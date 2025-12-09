use crate::{
    components::{partition::ReducePartition, precision::ReducePrecision},
    routines::ReduceBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub fn partition_parallel<P: ReducePrecision, Out: Numeric>(
    reduce_index: u32,
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    #[comptime] params: ReduceBlueprint,
) -> ReducePartition {
    let shape_axis = input.shape(axis_reduce);

    let mut index_start = 0;
    for axis in 0..input.rank() {
        let coordinate = output.coordinate(reduce_index, axis);
        index_start += coordinate * input.stride(axis);
    }
    index_start /= params.line_size_input;

    let coordinate_end = shape_axis;

    let coordinate_step = if params.shared.is_some() {
        CUBE_DIM * params.line_size_input
    } else if params.use_planes {
        CUBE_DIM_X * params.line_size_input
    } else {
        params.line_size_input.runtime()
    };

    ReducePartition {
        index_start,
        index_step: 1,
        coordinate_start: 0,
        coordinate_end,
        coordinate_step,
    }
}
