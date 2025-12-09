use crate::{BoundChecksInner, LineMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceBlueprint {
    pub shared: Option<u32>, // shared if Some(x) where x is the accumulator size.
    pub use_planes: bool,
    pub line_size_input: u32,
    pub line_size_output: u32,
    pub line_mode: LineMode,
    pub bound_checks: bool,
    pub bound_checks_inner: BoundChecksInner,
}
