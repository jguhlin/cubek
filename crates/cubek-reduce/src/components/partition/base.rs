use crate::{
    LineMode,
    components::{
        partition::{parallel::partition_parallel, perpendicular::partition_perpendicular},
        precision::ReducePrecision,
    },
    routines::ReduceBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

/// A simple range to specify how to iterate a slice when performing a reduction.
#[derive(CubeType)]
pub struct ReducePartition {
    pub index_start: u32,
    pub index_step: u32,
    pub coordinate_start: u32,
    pub coordinate_end: u32,
    pub coordinate_step: u32,
}

#[cube]
impl ReducePartition {
    pub(crate) fn new<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] params: ReduceBlueprint,
    ) -> ReducePartition {
        match comptime!(params.line_mode) {
            LineMode::Parallel => {
                partition_parallel::<P, Out>(reduce_index, input, output, axis_reduce, params)
            }
            LineMode::Perpendicular => {
                partition_perpendicular::<P, Out>(reduce_index, input, output, axis_reduce, params)
            }
        }
    }
}
