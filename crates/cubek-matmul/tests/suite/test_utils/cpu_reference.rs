use cubecl::{flex32, prelude::Numeric, tf32};
use cubek_matmul::components::{MatmulIdent, MatmulProblem};

use crate::suite::layered::matmul_test_launcher::strides;

pub trait CastInto<E> {
    fn cast_into(self) -> E;
}

impl<E> CastInto<E> for E {
    fn cast_into(self) -> E {
        self
    }
}

impl CastInto<f32> for half::f16 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<f32> for half::bf16 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<f32> for flex32 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<half::bf16> for f32 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self)
    }
}

impl CastInto<half::bf16> for half::f16 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self.to_f32())
    }
}

impl CastInto<half::f16> for half::bf16 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self.to_f32())
    }
}

impl CastInto<half::f16> for f32 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self)
    }
}

impl CastInto<half::f16> for flex32 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self.to_f32())
    }
}

impl CastInto<half::bf16> for flex32 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self.to_f32())
    }
}

impl CastInto<flex32> for f32 {
    fn cast_into(self) -> flex32 {
        flex32::from_f32(self)
    }
}

impl CastInto<f32> for tf32 {
    fn cast_into(self) -> f32 {
        self.to_f32()
    }
}

impl CastInto<tf32> for f32 {
    fn cast_into(self) -> tf32 {
        tf32::from_f32(self)
    }
}

impl CastInto<u16> for u8 {
    fn cast_into(self) -> u16 {
        self as u16
    }
}

impl CastInto<i32> for u16 {
    fn cast_into(self) -> i32 {
        self as i32
    }
}

impl CastInto<u8> for i32 {
    fn cast_into(self) -> u8 {
        self as u8
    }
}

/// Solves a matmul problem with EG inputs, multiplied as ES and accumulated as EA.
///
/// This is a naive CPU implementation, very slow on large payloads,
/// not designed to be used for other purposes than testing.
pub(crate) fn matmul_cpu_reference<
    EG: Numeric + CastInto<ES>,
    ES: Numeric + CastInto<EA>,
    EA: Numeric + CastInto<EG>,
>(
    lhs: &[EG],
    rhs: &[EG],
    problem: &MatmulProblem,
) -> Vec<EG>
where
{
    let m = problem.m;
    let n = problem.n;
    let k = problem.k;
    let num_batches = problem.num_batches();

    let b_lhs = problem.lhs_batches.clone();
    let b_rhs = problem.rhs_batches.clone();
    assert!(
        b_lhs.len() == b_rhs.len(),
        "Cpu reference only works with batches of equal length. Please pad the shortest one with ones at the beginning."
    );

    let lhs_strides = strides(problem, MatmulIdent::Lhs);
    let rhs_strides = strides(problem, MatmulIdent::Rhs);
    let out_strides = strides(problem, MatmulIdent::Out);

    let mut acc = vec![EA::from_int(0); m * n * num_batches];

    for nth_batch in 0..num_batches {
        let batch_out = nth_batch * m * n;
        let mut batch_lhs = 0;
        let mut batch_rhs = 0;
        for b in 0..b_lhs.len() {
            let tmp = batch_out / out_strides[b];
            batch_lhs += tmp % b_lhs[b] * lhs_strides[b];
            batch_rhs += tmp % b_rhs[b] * rhs_strides[b];
        }

        for i in 0..m {
            for j in 0..n {
                for k_ in 0..k {
                    let lhs_index = i * k + k_;
                    let rhs_index = k_ * n + j;
                    let out_index = i * n + j;

                    let l: ES = lhs[batch_lhs + lhs_index].cast_into();
                    let r: ES = rhs[batch_rhs + rhs_index].cast_into();
                    let prod = l * r;

                    acc[batch_out + out_index] += prod.cast_into();
                }
            }
        }
    }

    // Allows EG != EA
    if core::any::TypeId::of::<EG>() == core::any::TypeId::of::<EA>() {
        // EG == EA → return `acc` directly
        let acc_as_eg: Vec<EG> = unsafe { std::mem::transmute(acc) };
        acc_as_eg
    } else {
        // EG != EA → cast each element
        let mut out = vec![EG::from_int(0); m * n * num_batches];
        for i in 0..m * n * num_batches {
            out[i] = acc[i].cast_into();
        }
        out
    }
}
