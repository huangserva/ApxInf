//! Checkpoint loading for the X-VLA DaViT vision tower.
//!
//! The checkpoint is a Florence-2 layout (`vlm.vision_tower.*`). Dense
//! convolutions are transposed into im2col matrices on the host, linear layers
//! into `[input, output]` row-major form, and everything is uploaded once as
//! BF16 device tensors.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};
use half::bf16;

use crate::xvla::backend::kernels::{elementwise, xvla_ops::prepare_row_gather};
use crate::xvla::backend::{transfers, Context};
use crate::xvla::config::XvlaConfig;
use crate::xvla::math::image_position_table;
use crate::xvla::weights::{
    DepthwiseWeights, LayerNormWeights, LinearWeights, StagePlan, TextLayerWeights, TextWeights,
    VisionBlockWeights, VisionStageWeights, VisionWeights, VlmWeights,
};

impl VisionWeights {
    /// Load the vision tower from an X-VLA checkpoint directory or file.
    pub fn from_checkpoint(config: &XvlaConfig, path: &Path, ctx: &Context) -> Result<Self> {
        let (mut tensors, _) =
            apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
        load_vision(&mut tensors, config, ctx)
    }
}

impl TextWeights {
    /// Load the BART encoder from an X-VLA checkpoint directory or file.
    pub fn from_checkpoint(config: &XvlaConfig, path: &Path, ctx: &Context) -> Result<Self> {
        let (mut tensors, _) =
            apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
        load_text(&mut tensors, config, ctx)
    }
}

impl VlmWeights {
    /// Load vision and text weights from a single checkpoint pass.
    pub fn from_checkpoint(config: &XvlaConfig, path: &Path, ctx: &Context) -> Result<Self> {
        let (mut tensors, _) =
            apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
        let vision = load_vision(&mut tensors, config, ctx)?;
        let text = load_text(&mut tensors, config, ctx)?;
        Ok(Self { vision, text })
    }
}

fn load_vision(
    tensors: &mut HashMap<String, Tensor>,
    config: &XvlaConfig,
    ctx: &Context,
) -> Result<VisionWeights> {
    let vision = &config.vision;
    let side = config.profile.image_size;
    let sides = vision.feature_sides(side);

    let mut stages = Vec::with_capacity(4);
    let mut in_channels = 3usize;
    for stage in 0..4 {
        let channels = vision.dims[stage];
        let kernel = vision.patch_size[stage];
        let stride = vision.patch_stride[stage];
        let padding = vision.patch_padding[stage];
        let pre_norm = vision.patch_prenorm[stage];
        let prefix = format!("vlm.vision_tower.convs.{stage}");

        let conv_weight = take_f32(
            tensors,
            &format!("{prefix}.proj.weight"),
            &[channels, in_channels, kernel, kernel],
        )?;
        let conv_weight = device_bf16(
            ctx,
            &[in_channels * kernel * kernel, channels],
            &dense_conv_weight(&conv_weight, in_channels, channels, kernel),
        )?;
        let conv = LinearWeights {
            weight: conv_weight,
            bias: device_bf16(
                ctx,
                &[channels],
                &take_f32(tensors, &format!("{prefix}.proj.bias"), &[channels])?,
            )?,
        };

        let norm_dim = if pre_norm { in_channels } else { channels };
        let norm = take_layer_norm(ctx, tensors, &format!("{prefix}.norm"), norm_dim)?;

        let mut blocks = Vec::with_capacity(vision.depths[stage]);
        for pair in 0..vision.depths[stage] {
            let spatial = take_block(
                ctx,
                tensors,
                &format!("vlm.vision_tower.blocks.{stage}.{pair}"),
                "spatial_block",
                "window_attn",
                channels,
                vision.mlp_ratio,
            )?;
            let channel = take_block(
                ctx,
                tensors,
                &format!("vlm.vision_tower.blocks.{stage}.{pair}"),
                "channel_block",
                "channel_attn",
                channels,
                vision.mlp_ratio,
            )?;
            blocks.push((spatial, channel));
        }

        stages.push(VisionStageWeights {
            pre_norm,
            channels,
            heads: vision.num_heads[stage],
            groups: vision.num_groups[stage],
            conv,
            norm,
            blocks,
            plan: stage_plan(ctx, sides[stage], vision.window_size)?,
            zero_row: device_bf16(ctx, &[1, channels], &vec![0.0; channels])?,
            kernel,
            stride,
            padding,
        });
        in_channels = channels;
    }

    let half = vision.dims[3] / 2;
    let rows = take_f32(
        tensors,
        "vlm.image_pos_embed.row_embeddings.weight",
        &[vision.image_pos_max, half],
    )?;
    let columns = take_f32(
        tensors,
        "vlm.image_pos_embed.column_embeddings.weight",
        &[vision.image_pos_max, half],
    )?;
    let table = image_position_table(&rows, &columns, sides[3], half);

    let projected = vision.projection_dim;
    let image_projection = take_f32(
        tensors,
        "vlm.image_projection",
        &[vision.dims[3], projected],
    )?;
    let image_projection = device_bf16(ctx, &[vision.dims[3], projected], &image_projection)?;
    let image_proj_norm = take_layer_norm(ctx, tensors, "vlm.image_proj_norm", projected)?;

    let tokens = sides[3] * sides[3];
    Ok(VisionWeights {
        stages,
        image_position_table: device_bf16(ctx, &[tokens, half * 2], &table)?,
        image_projection,
        image_proj_norm,
        spatial_mean_row: device_bf16(ctx, &[1, tokens], &vec![1.0; tokens])?,
        input_side: side,
    })
}

fn load_text(
    tensors: &mut HashMap<String, Tensor>,
    config: &XvlaConfig,
    ctx: &Context,
) -> Result<TextWeights> {
    let text = &config.text;
    let prefix = "vlm.language_model.model";
    let d_model = text.d_model;
    let vocab = text.vocab_size;

    let embedding = device_bf16(
        ctx,
        &[vocab, d_model],
        &take_f32(
            tensors,
            &format!("{prefix}.shared.weight"),
            &[vocab, d_model],
        )?,
    )?;

    let position_rows = text.max_positions + text.pad_token_id + 1;
    let position_table = take_f32(
        tensors,
        &format!("{prefix}.encoder.embed_positions.weight"),
        &[position_rows, d_model],
    )?;
    let position_table = device_bf16(ctx, &[position_rows, d_model], &position_table)?;
    let image_tokens = config.vision.feature_sides(config.profile.image_size)[3].pow(2) + 1;
    let merged_tokens = image_tokens + config.profile.tokens;
    let rows: Vec<usize> = (text.position_offset..text.position_offset + merged_tokens).collect();
    let position_indices = elementwise::prepare_row_indices(ctx, &rows, position_rows)?;
    let positions =
        elementwise::gather_rows_bf16_prepared(ctx, &position_table, &position_indices)?;

    let embed_norm = take_layer_norm(
        ctx,
        tensors,
        &format!("{prefix}.encoder.layernorm_embedding"),
        d_model,
    )?;

    let head_dim = d_model / text.num_heads;
    let mut layers = Vec::with_capacity(text.layers);
    for index in 0..text.layers {
        let layer = format!("{prefix}.encoder.layers.{index}");
        layers.push(TextLayerWeights {
            q_proj: take_linear(
                ctx,
                tensors,
                &format!("{layer}.self_attn.q_proj"),
                d_model,
                d_model,
            )?,
            k_proj: take_linear(
                ctx,
                tensors,
                &format!("{layer}.self_attn.k_proj"),
                d_model,
                d_model,
            )?,
            v_proj: take_linear(
                ctx,
                tensors,
                &format!("{layer}.self_attn.v_proj"),
                d_model,
                d_model,
            )?,
            out_proj: take_linear(
                ctx,
                tensors,
                &format!("{layer}.self_attn.out_proj"),
                d_model,
                d_model,
            )?,
            attn_norm: take_layer_norm(
                ctx,
                tensors,
                &format!("{layer}.self_attn_layer_norm"),
                d_model,
            )?,
            fc1: take_linear(ctx, tensors, &format!("{layer}.fc1"), d_model, text.ffn_dim)?,
            fc2: take_linear(ctx, tensors, &format!("{layer}.fc2"), text.ffn_dim, d_model)?,
            ffn_norm: take_layer_norm(ctx, tensors, &format!("{layer}.final_layer_norm"), d_model)?,
        });
    }

    Ok(TextWeights {
        embedding,
        positions,
        embed_norm,
        layers,
        heads: text.num_heads,
        head_dim,
    })
}

#[allow(clippy::too_many_arguments)]
fn take_block(
    ctx: &Context,
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    block: &str,
    attention: &str,
    channels: usize,
    mlp_ratio: usize,
) -> Result<VisionBlockWeights> {
    let block = format!("{prefix}.{block}");
    let attention = format!("{block}.{attention}");
    let hidden = channels * mlp_ratio;
    Ok(VisionBlockWeights {
        conv1: take_depthwise(ctx, tensors, &format!("{block}.conv1.fn.dw"), channels)?,
        norm_attn: take_layer_norm(ctx, tensors, &format!("{attention}.norm"), channels)?,
        qkv: take_linear(
            ctx,
            tensors,
            &format!("{attention}.fn.qkv"),
            channels,
            channels * 3,
        )?,
        proj: take_linear(
            ctx,
            tensors,
            &format!("{attention}.fn.proj"),
            channels,
            channels,
        )?,
        conv2: take_depthwise(ctx, tensors, &format!("{block}.conv2.fn.dw"), channels)?,
        norm_ffn: take_layer_norm(ctx, tensors, &format!("{block}.ffn.norm"), channels)?,
        fc1: take_linear(
            ctx,
            tensors,
            &format!("{block}.ffn.fn.net.fc1"),
            channels,
            hidden,
        )?,
        fc2: take_linear(
            ctx,
            tensors,
            &format!("{block}.ffn.fn.net.fc2"),
            hidden,
            channels,
        )?,
    })
}

fn take_layer_norm(
    ctx: &Context,
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    dim: usize,
) -> Result<LayerNormWeights> {
    Ok(LayerNormWeights {
        weight: device_bf16(
            ctx,
            &[dim],
            &take_f32(tensors, &format!("{prefix}.weight"), &[dim])?,
        )?,
        bias: device_bf16(
            ctx,
            &[dim],
            &take_f32(tensors, &format!("{prefix}.bias"), &[dim])?,
        )?,
    })
}

fn take_linear(
    ctx: &Context,
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    input_dim: usize,
    output_dim: usize,
) -> Result<LinearWeights> {
    let weight = take_f32(
        tensors,
        &format!("{prefix}.weight"),
        &[output_dim, input_dim],
    )?;
    Ok(LinearWeights {
        weight: device_bf16(
            ctx,
            &[input_dim, output_dim],
            &transpose_2d(&weight, output_dim, input_dim),
        )?,
        bias: device_bf16(
            ctx,
            &[output_dim],
            &take_f32(tensors, &format!("{prefix}.bias"), &[output_dim])?,
        )?,
    })
}

fn take_depthwise(
    ctx: &Context,
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    channels: usize,
) -> Result<DepthwiseWeights> {
    let weight = take_f32(tensors, &format!("{prefix}.weight"), &[channels, 1, 3, 3])?;
    Ok(DepthwiseWeights {
        weight: device_bf16(ctx, &[channels, 9], &weight)?,
        bias: device_bf16(
            ctx,
            &[channels],
            &take_f32(tensors, &format!("{prefix}.bias"), &[channels])?,
        )?,
    })
}

/// Padded window partition and its inverse for one feature-map side.
fn stage_plan(ctx: &Context, side: usize, window: usize) -> Result<StagePlan> {
    if side == 0 || window == 0 {
        return Err(Error::Other(
            "X-VLA stage plan needs a positive side".into(),
        ));
    }
    let padded = side.div_ceil(window) * window;
    let windows_per_side = padded / window;
    let windows = windows_per_side * windows_per_side;
    let footprint = window * window;
    let mut partition = Vec::with_capacity(windows * footprint);
    for window_y in 0..windows_per_side {
        for window_x in 0..windows_per_side {
            for y in 0..window {
                for x in 0..window {
                    let (iy, ix) = (window_y * window + y, window_x * window + x);
                    partition.push(if iy < side && ix < side {
                        (iy * side + ix) as u32
                    } else {
                        (side * side) as u32
                    });
                }
            }
        }
    }
    let mut reverse = vec![0u32; side * side];
    for (slot, token) in partition.iter().enumerate() {
        let token = *token as usize;
        if token < side * side {
            reverse[token] = slot as u32;
        }
    }
    Ok(StagePlan {
        height: side,
        width: side,
        window,
        windows,
        partition: prepare_row_gather(ctx, &partition, side * side + 1)?,
        reverse: prepare_row_gather(ctx, &reverse, windows * footprint)?,
    })
}

fn take_f32(tensors: &mut HashMap<String, Tensor>, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing X-VLA weight `{name}`")))?;
    if tensor.shape().dims() != dims {
        return Err(Error::Other(format!(
            "X-VLA weight `{name}` has shape {:?}, expected {dims:?}",
            tensor.shape().dims()
        )));
    }
    tensor_values(&tensor)
}

fn tensor_values(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(match tensor.dtype() {
        DType::F32 => tensor.as_f32()?.to_vec(),
        DType::F16 => tensor
            .as_f16()?
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        DType::BF16 => tensor
            .as_bf16()?
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        other => {
            return Err(Error::Other(format!(
                "X-VLA checkpoint weight has unsupported dtype {other}"
            )))
        }
    })
}

fn device_bf16(ctx: &Context, dims: &[usize], values: &[f32]) -> Result<Tensor> {
    let rounded: Vec<bf16> = values.iter().map(|value| bf16::from_f32(*value)).collect();
    let host = Tensor::from_bf16(dims.to_vec(), &rounded)?;
    transfers::to_cuda(&host, ctx.device_id())
}

fn transpose_2d(source: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut target = vec![0.0f32; source.len()];
    for row in 0..rows {
        for col in 0..cols {
            target[col * rows + row] = source[row * cols + col];
        }
    }
    target
}

/// `[out, in, k, k]` → `[in * k * k, out]` in im2col tap order.
fn dense_conv_weight(
    source: &[f32],
    in_channels: usize,
    out_channels: usize,
    k: usize,
) -> Vec<f32> {
    let mut target = vec![0.0f32; source.len()];
    for out in 0..out_channels {
        for channel in 0..in_channels {
            for ky in 0..k {
                for kx in 0..k {
                    let from = ((out * in_channels + channel) * k + ky) * k + kx;
                    let to = ((channel * k + ky) * k + kx) * out_channels + out;
                    target[to] = source[from];
                }
            }
        }
    }
    target
}
