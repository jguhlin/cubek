use crate::suite::TestEG;
use crate::suite::test_utils::{assert_result, tensor_raw_parts};
use cubecl::Runtime;
use cubecl::frontend::CubePrimitive;
use cubecl::prelude::TensorHandleRef;
use cubek_matmul::MatmulInputHandleRef;

use cubek_matmul::components::{MatmulIdent, MatmulProblem, MatrixLayout};
use cubek_matmul::{components::MatmulElems, kernels::naive};

type TestRuntime = cubecl::TestRuntime;

struct MatmulTestCase {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
}

impl MatmulTestCase {
    fn to_problem(self) -> MatmulProblem {
        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            lhs_batches: vec![self.batch],
            rhs_batches: vec![self.batch],
            out_batches: vec![self.batch],
            lhs_strides: vec![self.m * self.k, self.k],
            rhs_strides: vec![self.k * self.n, self.n],
            lhs_layout: MatrixLayout::RowMajor,
            rhs_layout: MatrixLayout::RowMajor,
        }
    }
}

#[test]
pub fn test_small() {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
    };

    test_naive(case);
}

#[test]
pub fn test_odd() {
    let case = MatmulTestCase {
        m: 1,
        k: 101,
        n: 255,
        batch: 1,
    };

    test_naive(case);
}

#[test]
pub fn test_large() {
    let case = MatmulTestCase {
        m: 256,
        k: 256,
        n: 256,
        batch: 1,
    };

    test_naive(case);
}

#[test]
pub fn test_with_check_bounds() {
    let case = MatmulTestCase {
        m: 60,
        k: 60,
        n: 60,
        batch: 1,
    };

    test_naive(case);
}

#[test]
pub fn test_with_batches() {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 3,
    };

    test_naive(case);
}

fn test_naive(case: MatmulTestCase) {
    let client = TestRuntime::client(&Default::default());
    let problem = case.to_problem();

    let lhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Lhs);
    let rhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Rhs);
    let out = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Out);

    let lhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(
                &lhs.handle,
                &lhs.strides,
                &lhs.shape,
                TestEG::type_size() as usize,
            )
        },
        TestEG::as_type_native_unchecked(),
    );
    let rhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(
                &rhs.handle,
                &rhs.strides,
                &rhs.shape,
                TestEG::type_size() as usize,
            )
        },
        TestEG::as_type_native_unchecked(),
    );
    let out_handle = unsafe {
        TensorHandleRef::from_raw_parts(
            &out.handle,
            &out.strides,
            &out.shape,
            TestEG::type_size() as usize,
        )
    };

    naive::launch_ref(
        &client,
        &lhs_handle,
        &rhs_handle,
        &out_handle,
        &MatmulElems::new::<TestEG>(),
    )
    .unwrap();

    assert_result(
        &lhs.original_data.unwrap(),
        &rhs.original_data.unwrap(),
        &problem,
        &client,
        out.handle,
        &out.shape,
        &out.strides,
    );
}
