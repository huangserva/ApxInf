//! X-VLA runtime: observation binding, RGB preprocessing, and eager execution.
//!
//! The runtime is the VLA-family seam: it turns one [`VlaRequest`] into the
//! three-stage forward pass (DaViT view tower, Florence-2 merged encoder, linear
//! flow action head). RGB frames are normalized and patchified on device by the
//! X-VLA kernels, so no host-side pixel preprocessing happens here.
//!
//! Preparation is eager in this step. CUDA graph capture of the three stages is
//! a separate execution-policy step; `PreferGraph` therefore reports an explicit
//! fallback instead of silently claiming a graph.

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, SamplingBackend, Shape, Tensor};
use half::bf16;

use crate::accelerator::cuda::RuntimeBackend;
use crate::auto::{LoadOptions, LoadedModel};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    PreparationStatus, PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};

use super::backend::kernels::xvla_ops;
use super::backend::{transfers, Context, DeviceBuffer};
use super::config::XvlaConfig;
use super::model::action::generate_actions;
use super::model::text::encode_vlm;
use super::model::vision::encode_image;
use super::weights::{ActionHeadWeights, VlmWeights};

pub(crate) fn register_builtin() {
    crate::registry::register("xvla-cuda", load_registered);
}

fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    Ok(LoadedModel::Vla(Box::new(load_runtime(path, backend)?)))
}

fn load_runtime(path: &Path, backend: Arc<dyn Backend>) -> Result<XvlaRuntime> {
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("X-VLA is only registered for CUDA".into()))?;
    let root = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };
    let config = Arc::new(XvlaConfig::from_checkpoint(&root)?);
    let ctx = backend.context();
    let vlm = Arc::new(VlmWeights::from_checkpoint(&config, path, ctx)?);
    let head = Arc::new(ActionHeadWeights::from_checkpoint(&config, path, ctx)?);
    Ok(XvlaRuntime {
        backend,
        config,
        vlm,
        head,
    })
}

/// Loaded X-VLA checkpoint plus its execution resources.
pub(crate) struct XvlaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<XvlaConfig>,
    vlm: Arc<VlmWeights>,
    head: Arc<ActionHeadWeights>,
}

impl VlaRuntime for XvlaRuntime {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }

    fn contract(&self) -> VlaContract {
        let profile = &self.config.profile;
        VlaContract {
            action_shape: [profile.action_horizon, self.config.action_space.dim_action],
            // This runtime accepts raw RGB only; the patch row count is not a
            // fixed part of its wire contract.
            patch_shape: [0, 3],
            max_token_len: profile.tokens,
            num_views: profile.num_views,
            image_size: profile.image_size,
            // Input pixels covered by one stage-4 DaViT token (strides 4,2,2,2).
            patch_size: profile.image_size
                / self.config.vision.feature_sides(profile.image_size)[3],
            accepts_rgb_u8: true,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        run_xvla(&self.backend, &self.config, &self.vlm, &self.head, request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::PreferGraph)
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        let fallback_reason = match policy {
            ExecutionPolicy::Eager => None,
            ExecutionPolicy::PreferGraph => {
                Some("X-VLA CUDA graph capture is not implemented yet; running eager".into())
            }
            ExecutionPolicy::RequireGraph => {
                return Err(Error::Other(
                    "X-VLA CUDA graph capture is not implemented yet".into(),
                ))
            }
        };
        Ok(Box::new(XvlaPreparedInference {
            spec: *spec,
            backend: self.backend.clone(),
            config: self.config.clone(),
            vlm: self.vlm.clone(),
            head: self.head.clone(),
            fallback_reason,
        }))
    }

    fn execution_mode(&self) -> &'static str {
        "eager"
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        transfers::to_cpu(action.tensor())?.to_f32_vec()
    }
}

/// Eager fixed-spec plan. Owns the shared runtime resources so it can outlive
/// the runtime handle that produced it.
struct XvlaPreparedInference {
    spec: InferenceSpec,
    backend: Arc<RuntimeBackend>,
    config: Arc<XvlaConfig>,
    vlm: Arc<VlmWeights>,
    head: Arc<ActionHeadWeights>,
    fallback_reason: Option<String>,
}

impl PreparedInference for XvlaPreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: ExecutionMode::Eager,
            fallback_reason: self.fallback_reason.clone(),
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(format!(
                "prepared X-VLA spec {:?} does not match observation {:?}",
                self.spec,
                request.observation.inference_spec()
            )));
        }
        run_xvla(&self.backend, &self.config, &self.vlm, &self.head, request)
    }
}

fn run_xvla(
    backend: &RuntimeBackend,
    config: &XvlaConfig,
    vlm: &VlmWeights,
    head: &ActionHeadWeights,
    request: &VlaRequest<'_>,
) -> Result<Action> {
    let observation = request.observation;
    observation.validate()?;
    let ctx = backend.context();
    let profile = &config.profile;
    let side = profile.image_size;
    if observation.token_ids.len() != profile.tokens {
        return Err(Error::Other(format!(
            "X-VLA expects {} instruction tokens, got {}",
            profile.tokens,
            observation.token_ids.len()
        )));
    }
    let bytes = match &observation.vision {
        VisionObservation::RgbU8 { bytes, layout } => {
            if *layout != ImageLayout::Nhwc {
                return Err(Error::Other(
                    "X-VLA RGB input must be NHWC; transpose channels-first frames in the policy"
                        .into(),
                ));
            }
            bytes
        }
        VisionObservation::Patches(_) => {
            return Err(Error::Other(
                "X-VLA does not accept preprocessed patches; supply raw RGB frames".into(),
            ))
        }
    };
    let frame_bytes = side * side * 3;
    if bytes.is_empty() || bytes.len() % frame_bytes != 0 {
        return Err(Error::Other(format!(
            "X-VLA RGB input must contain whole {side}x{side}x3 frames, got {} bytes",
            bytes.len()
        )));
    }
    let views = bytes.len() / frame_bytes;
    if views > profile.num_views {
        return Err(Error::Other(format!(
            "X-VLA accepts at most {} camera views, got {views}",
            profile.num_views
        )));
    }
    let state = observation
        .state
        .as_ref()
        .ok_or_else(|| Error::Other("X-VLA requires proprioception (Observation.state)".into()))?;
    let proprio = upload_bf16(
        ctx,
        state,
        &[config.action_space.dim_proprio],
        "proprioception",
    )?;
    let domain = request.metadata.embodiment_id.ok_or_else(|| {
        Error::Other("X-VLA requires a domain id (VlaMetadata.embodiment_id)".into())
    })?;
    if domain >= config.head.num_domains {
        return Err(Error::Other(format!(
            "X-VLA domain {domain} is out of range (domains = {})",
            config.head.num_domains
        )));
    }
    let latent_shape = [profile.action_horizon, config.action_space.dim_action];
    let latent = match request.initial_latent {
        InitialLatent::Provided(tensor) => {
            upload_bf16(ctx, tensor, &latent_shape, "initial latent")?
        }
        InitialLatent::Generate { rng } => {
            let elements = latent_shape[0] * latent_shape[1];
            let buffer = DeviceBuffer::alloc_on(ctx, elements * DType::BF16.size_in_bytes())
                .map_err(Error::Cuda)?;
            let noise = buffer
                .as_tensor(Shape::new(latent_shape.to_vec()), DType::BF16)
                .map_err(Error::Cuda)?;
            let mut generator = backend.create_normal_generator(noise.clone())?;
            generator.generate(rng)?;
            noise
        }
    };

    let mut features: Vec<Option<Tensor>> = Vec::with_capacity(profile.num_views);
    for index in 0..views {
        let frame = &bytes[index * frame_bytes..(index + 1) * frame_bytes];
        let raw = xvla_ops::upload_bytes(ctx, frame)?;
        let tokens = xvla_ops::rgb_u8_to_normalized_tokens_bf16(
            ctx,
            &raw,
            side,
            side,
            xvla_ops::XVLA_IMAGE_MEAN,
            xvla_ops::XVLA_IMAGE_STD,
        )?;
        features.push(Some(encode_image(
            ctx,
            &vlm.vision,
            &tokens,
            config.layer_norm_eps,
        )?));
    }
    while features.len() < profile.num_views {
        features.push(None);
    }
    let refs: Vec<Option<&Tensor>> = features.iter().map(Option::as_ref).collect();
    let output = encode_vlm(
        ctx,
        vlm,
        &refs,
        &observation.token_ids,
        config.layer_norm_eps,
    )?;
    let action = generate_actions(
        ctx,
        head,
        config,
        &output.vlm_features,
        &output.aux_visual_inputs,
        &proprio,
        domain,
        &latent,
    )?;
    Ok(Action::new(action))
}

/// Convert one host float tensor of `dims` into device BF16 storage.
fn upload_bf16(ctx: &Context, tensor: &Tensor, dims: &[usize], label: &str) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!("X-VLA {label} must be a CPU tensor")));
    }
    if tensor.shape().dims() != dims {
        return Err(Error::Other(format!(
            "X-VLA {label} shape {:?} does not match {dims:?}",
            tensor.shape().dims()
        )));
    }
    let host = match tensor.dtype() {
        DType::BF16 => tensor.clone(),
        DType::F32 | DType::F16 => {
            let rounded: Vec<bf16> = tensor
                .to_f32_vec()?
                .into_iter()
                .map(bf16::from_f32)
                .collect();
            Tensor::from_bf16(dims.to_vec(), &rounded)?
        }
        dtype => {
            return Err(Error::Other(format!(
                "X-VLA {label} must be f32, f16 or bf16, got {dtype}"
            )))
        }
    };
    transfers::to_cuda(&host, ctx.device_id())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use apxinf_core::{Device, Shape, Tensor};

    use super::load_runtime;
    use crate::vla::{
        ImageLayout, Observation, VisionObservation, VlaMetadata, VlaRequest, VlaRuntime,
    };

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn read_i64(path: &Path) -> Vec<i64> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn pixels_to_rgb_u8(pixels: &[f32], side: usize) -> Vec<u8> {
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut out = Vec::with_capacity(side * side * 3);
        for position in 0..side * side {
            for channel in 0..3 {
                let value = pixels[channel * side * side + position] * std[channel] + mean[channel];
                out.push((value * 255.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    /// Full runtime binding: RGB frames -> normalize -> DaViT -> BART -> flow.
    /// The oracle stores normalized fp32 pixels; this harness quantizes them
    /// back to u8, so the gate is wider than the head-only parity test.
    #[test]
    #[ignore = "GPU parity harness: set XVLA_RUNTIME_CASE"]
    fn runtime_matches_oracle() {
        let case = std::env::var("XVLA_RUNTIME_CASE").expect("XVLA_RUNTIME_CASE is required");
        let case = Path::new(&case);
        let tolerance: f32 = std::env::var("XVLA_RUNTIME_TOL")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.02);
        let checkpoint = std::env::var("XVLA_CKPT_DIR").unwrap_or_else(|_| {
            format!(
                "{}/refs/checkpoints/X-VLA-Libero",
                std::env::var("HOME").expect("HOME")
            )
        });
        let backend = crate::accelerator::create_backend(Device::Cuda(0)).unwrap();
        let runtime = load_runtime(Path::new(&checkpoint), backend).unwrap();
        let side = runtime.contract().image_size;

        let mask = std::fs::read(case.join("view_mask.u8")).unwrap();
        let mut rgb = Vec::new();
        let mut valid = 0usize;
        for present in mask {
            if present != 0 {
                let pixels = read_f32(&case.join(format!("pixel_{valid}.f32")));
                rgb.extend(pixels_to_rgb_u8(&pixels, side));
                valid += 1;
            }
        }
        let ids: Vec<u32> = read_i64(&case.join("input_ids.i64"))
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        let proprio =
            Tensor::from_f32(Shape::new(vec![20]), &read_f32(&case.join("proprio.f32"))).unwrap();
        let x1 =
            Tensor::from_f32(Shape::new(vec![30, 20]), &read_f32(&case.join("x1.f32"))).unwrap();
        let domain = read_i64(&case.join("domain_id.i64"))[0] as usize;
        let observation = Observation {
            vision: VisionObservation::RgbU8 {
                bytes: rgb,
                layout: ImageLayout::Nhwc,
            },
            token_ids: ids,
            state: Some(proprio),
            action_mask: None,
        };
        let metadata = VlaMetadata {
            embodiment_id: Some(domain),
            ..VlaMetadata::default()
        };
        let request = VlaRequest::provided_with_metadata(&observation, &x1, metadata);
        let actual = runtime.infer_host_f32(&request).unwrap();
        let expected = read_f32(&case.join("final_action.f32"));
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut expected_norm = 0.0f64;
        for (actual, expected) in actual.iter().zip(&expected) {
            let difference = (actual - expected).abs();
            max_abs = max_abs.max(difference);
            sum_abs += difference as f64;
            dot += (*actual as f64) * (*expected as f64);
            actual_norm += (*actual as f64) * (*actual as f64);
            expected_norm += (*expected as f64) * (*expected as f64);
        }
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        println!(
            "X-VLA runtime_action parity: max_abs={max_abs:.6} mean_abs={:.6} cosine={cosine:.8}",
            sum_abs / actual.len() as f64
        );
        assert!(
            max_abs <= tolerance,
            "X-VLA runtime parity failed: max_abs={max_abs} > {tolerance}"
        );
        assert!(cosine >= 0.999, "X-VLA runtime cosine {cosine} below 0.999");
    }
}
