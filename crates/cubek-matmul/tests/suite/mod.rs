#![allow(missing_docs)]

// pub type TestEG = half::f16;
// pub type TestEG = f32;
// pub type TestES = f32;
// pub type TestEA = f32;
// TestEG: Float + CubeElement + Display + CastInto<TestES> + Sample + MatmulPrecision,
// TestES: Numeric + CastInto<TestEA>,
// TestEA: Numeric + CastInto<TestEG>,

pub mod layered;
pub mod naive;
pub mod test_utils;

// mod unit {
//     crate::testgen_matmul_unit!();
// }
// mod tma {
//     crate::testgen_matmul_tma!();
// }
