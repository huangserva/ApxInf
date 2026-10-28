//! Device-resident DaViT weights for the X-VLA vision tower.
//!
//! Every linear matrix is stored row-major `[input, output]` (checkpoint
//! `[output, input]` transposed at load time), dense convolution weights are
//! flattened in im2col tap order `[in_channels * kh * kw, out_channels]`, and
//! depthwise weights are `[channels, 9]` with `ky * 3 + kx` tap order.

use apxinf_core::Tensor;

use crate::xvla::backend::kernels::xvla_ops::RowGatherPlan;

#[derive(Debug)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct DepthwiseWeights {
    /// `[channels, 9]` in `ky * 3 + kx` tap order.
    pub weight: Tensor,
    pub bias: Tensor,
}

/// One DaViT block. Spatial and channel stages share this layout; the stage
/// plan decides which attention kernel runs.
#[derive(Debug)]
pub struct VisionBlockWeights {
    pub conv1: DepthwiseWeights,
    pub norm_attn: LayerNormWeights,
    pub qkv: LinearWeights,
    pub proj: LinearWeights,
    pub conv2: DepthwiseWeights,
    pub norm_ffn: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

/// Host-side window partition tables (device index buffers) for one stage.
pub struct StagePlan {
    pub height: usize,
    pub width: usize,
    pub window: usize,
    pub windows: usize,
    /// Padded window partition: `windows * window²` rows gathered from a
    /// `tokens + 1` matrix whose trailing row is all zeros.
    pub partition: RowGatherPlan,
    /// Inverse permutation back to `tokens` rows.
    pub reverse: RowGatherPlan,
}

pub struct VisionStageWeights {
    pub pre_norm: bool,
    pub channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub heads: usize,
    pub groups: usize,
    /// Dense patch-embedding convolution `[channels_in * k * k, channels]`.
    pub conv: LinearWeights,
    /// `pre_norm ? LayerNorm(channels_in) : LayerNorm(channels)`.
    pub norm: LayerNormWeights,
    /// `(spatial, channel)` block pairs in checkpoint order.
    pub blocks: Vec<(VisionBlockWeights, VisionBlockWeights)>,
    pub plan: StagePlan,
    /// `[1, channels]` zeros; padding sentinel for window gathers.
    pub zero_row: Tensor,
}

pub struct VisionWeights {
    /// Vision-tower input side (224 for the validated profile).
    pub input_side: usize,
    pub stages: Vec<VisionStageWeights>,
    /// `[49, 2048]`: column ‖ row embeddings plus the folded temporal constant.
    pub image_position_table: Tensor,
    /// `[2048, 1024]` stored as checkpoint `[in, out]` (no transpose).
    pub image_projection: Tensor,
    pub image_proj_norm: LayerNormWeights,
    /// `[1, 49]` of ones for the spatial average.
    pub spatial_mean_row: Tensor,
}
