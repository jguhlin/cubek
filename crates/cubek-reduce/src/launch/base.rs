use crate::{
    components::{
        args::{ReduceArgs, TensorArgs, init_tensors},
        instructions::*,
    },
    launch::{ReduceLaunchInfo, ReduceStrategy},
    routines::{ReduceBlueprint, reduce_kernel_virtual},
};
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ReduceDtypes {
    pub input: StorageType,
    pub output: StorageType,
    pub accumulation: StorageType,
}

/// Launch a reduce kernel. This function assumes that all parameters are already validated.
/// See the main entrypoint `reduce` in `lib.rs` for an example how to call this function
/// with the appropriate assumptions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_reduce<Run: Runtime>(
    client: &ComputeClient<Run>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    config: ReduceLaunchInfo,
    strategy: ReduceStrategy,
    dtypes: ReduceDtypes,
    inst: ReduceOperationConfig,
) -> Result<(), LaunchError> {
    let settings = ReduceBlueprint {
        shared: strategy.shared.then(|| {
            if strategy.use_planes {
                config.cube_dim.y
            } else {
                config.cube_dim.num_elems()
            }
        }),
        use_planes: strategy.use_planes,
        line_size_input: config.line_size_input,
        line_size_output: config.line_size_output,
        line_mode: config.line_mode,
        bound_checks: config.bound_checks,
        bound_checks_inner: config.bound_checks_inner,
    };
    unsafe {
        reduce_kernel::launch_unchecked::<TensorArgs, Run>(
            client,
            config.cube_count,
            config.cube_dim,
            input.as_tensor_arg(config.line_size_input as u8),
            output.as_tensor_arg(config.line_size_output as u8),
            ScalarArg::new(axis),
            settings,
            inst,
            dtypes.input,
            dtypes.output,
            dtypes.accumulation,
        )
    }
}

#[cube(launch_unchecked)]
pub fn reduce_kernel<In: Numeric, Out: Numeric, Acc: Numeric, RA: ReduceArgs>(
    input: &RA::Input<In>,
    output: &mut RA::Output<Out>,
    axis_reduce: u32,
    #[comptime] params: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
    #[define(In)] _input_dtype: StorageType,
    #[define(Out)] _output_dtype: StorageType,
    #[define(Acc)] _acc_dtype: StorageType,
) {
    let (input, mut output) = init_tensors::<RA, In, Out>(input, output);
    reduce_kernel_virtual::<In, Out, Acc>(&input, &mut output, axis_reduce, params, config);
}
