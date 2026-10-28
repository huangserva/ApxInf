//! Compile-time backend seam for X-VLA.
//!
//! Model code depends on this model-local alias and the model-neutral kernel
//! contract. Adding another accelerator backend changes this seam, not the
//! layer topology.

pub use crate::accelerator::cuda::kernels::preprocess::ImageLayout;
pub(crate) use crate::accelerator::cuda::{
    kernels, transfers, tuning, Context, DeviceBuffer, RuntimeBackend,
};
