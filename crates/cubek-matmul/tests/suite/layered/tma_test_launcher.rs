use cubecl::{TestRuntime, prelude::*};

use cubek_matmul::components::AvailableLineSizes;
use cubek_matmul::components::MatmulProblem;
use cubek_matmul::components::MatmulSelection;
use cubek_matmul::components::batch::BatchConfig;
use cubek_matmul::components::batch::BatchMatmulFamily;
use cubek_matmul::components::global::args::TensorMapArgs;
use cubek_matmul::components::global::args::{ConcreteInputsFactory, TensorMapInputs};
use cubek_matmul::components::{MatmulElems, MatmulIdent};
use cubek_matmul::kernels::layered::Algorithm;
use cubek_matmul::{
    MatmulInputHandleRef,
    components::global::args::{ConcreteOutputFactory, TensorOutput},
};

use crate::suite::test_utils::output_test_tensor;
use crate::suite::test_utils::{assert_result, input_test_tensor};

// TODO should be always used, remove feature flags
#[allow(unused)]
/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_tma_matmul_algorithm<A: Algorithm>(
    client: ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    selection: MatmulSelection,
    dtypes: MatmulElems,
) {
    let env = std::env::var("CUBEK_TEST_MODE");

    let panic_on_launch_err = match env {
        Ok(val) => match val.as_str() {
            "panic" => true,
            "skip" => false,
            _ => false,
        },
        Err(_) => false,
    };

    let line_sizes = AvailableLineSizes::from_type_sizes(
        &client,
        dtypes.lhs_global.size(),
        dtypes.rhs_global.size(),
        dtypes.acc_global.size(),
    );
    let line_sizes = A::filter_line_sizes(line_sizes);
    let line_sizes = line_sizes
        .filter_lhs(|ls| *ls == 1)
        .filter_rhs(|ls| *ls == 1)
        .pick_max()
        .unwrap();
    // let dtypes = MatmulElems::new_with_tile::<P::MP, A::TileMatmul>();
    let config = match A::setup(&client, &problem, &selection, &line_sizes, &dtypes) {
        Ok(config) => config,
        Err(err) => {
            let msg = format!("Can't launch the test: {err}");
            if panic_on_launch_err {
                panic!("{msg}");
            } else {
                println!("{msg}");
                return;
            }
        }
    };

    let line_sizes = config.line_sizes();

    let (lhs, lhs_data) = input_test_tensor(
        &client,
        dtypes.lhs_global,
        1234,
        problem.lhs_layout,
        problem.shape(MatmulIdent::Lhs),
    );
    let (rhs, rhs_data) = input_test_tensor(
        &client,
        dtypes.rhs_global,
        5678,
        problem.rhs_layout,
        problem.shape(MatmulIdent::Rhs),
    );
    let out = output_test_tensor(&client, &problem, dtypes.acc_global);

    problem.lhs_strides = lhs.strides.clone();
    problem.rhs_strides = rhs.strides.clone();

    let lhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(
                &lhs.handle,
                &lhs.strides,
                &lhs.shape,
                dtypes.lhs_global.size(),
            )
        },
        *dtypes.lhs_global,
    );
    let rhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(
                &rhs.handle,
                &rhs.strides,
                &rhs.shape,
                dtypes.rhs_global.size(),
            )
        },
        *dtypes.rhs_global,
    );
    let out_handle = unsafe {
        TensorHandleRef::from_raw_parts(
            &out.handle,
            &out.strides,
            &out.shape,
            dtypes.acc_global.size(),
        )
    };

    let inputs = TensorMapInputs::create(
        &client,
        &lhs_handle,
        &rhs_handle,
        &selection,
        &problem,
        &line_sizes,
        config,
        &dtypes,
    );
    let output = TensorOutput::create(
        &client,
        &out_handle,
        &selection,
        &problem,
        &line_sizes,
        config,
        &dtypes,
    );
    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    let result = unsafe {
        A::BatchMatmul::launch_unchecked::<TensorMapArgs, TestRuntime>(
            &client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            inputs,
            output,
            cube_count_plan.as_args(),
            config,
            &dtypes,
        )
    };

    match result {
        Ok(()) => {}
        Err(_err) => return,
    }

    assert_result(&lhs_data, &rhs_data, &problem, &client, &out, dtypes);
}
