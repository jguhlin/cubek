use std::any::TypeId;
use std::fmt::Display;

use cubecl::TestRuntime;
use cubecl::{CubeElement, client::ComputeClient, prelude::Float, server};

use crate::suite::test_utils::cpu_reference::matmul_cpu_reference;
use crate::suite::{TestEA, TestEG, TestES};
use cubek_matmul::components::MatmulProblem;

pub fn assert_result(
    lhs: &[TestEG],
    rhs: &[TestEG],
    problem: &MatmulProblem,
    client: &ComputeClient<TestRuntime>,
    out: server::Handle,
    shape: &[usize],
    strides: &[usize],
) {
    let eps_global = if TypeId::of::<TestEG>() == TypeId::of::<f32>() {
        f32::EPSILON
    } else {
        half::f16::EPSILON.to_f32()
    };
    let eps_stage = if TypeId::of::<TestES>() == TypeId::of::<f32>() {
        f32::EPSILON
    } else {
        half::f16::EPSILON.to_f32()
    };
    let eps_acc = if TypeId::of::<TestEA>() == TypeId::of::<f32>() {
        f32::EPSILON
    } else {
        half::f16::EPSILON.to_f32()
    };

    // Empirically found on metal
    let safety_factor = 150.0;
    let epsilon = (eps_global.max(eps_stage).max(eps_acc)) * safety_factor;

    let expected = matmul_cpu_reference::<TestEG, TestES, TestEA>(lhs, rhs, problem)
        .into_iter()
        .collect::<Vec<TestEG>>();

    if let Err(e) = assert_equals_approx::<TestEG>(client, out, shape, strides, &expected, epsilon)
    {
        panic!("{}", e);
    }
}

/// Compares the content of a handle to a given slice of f32.
fn assert_equals_approx<F: Float + CubeElement + Display>(
    client: &ComputeClient<TestRuntime>,
    output: server::Handle,
    shape: &[usize],
    strides: &[usize],
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let env = std::env::var("MATMUL_TEST_MODE");

    let print_instead_of_compare = match env {
        Ok(val) => matches!(val.as_str(), "print"),
        Err(_) => false,
    };

    let actual =
        client.read_one_tensor(output.copy_descriptor(shape, strides, F::type_size() as usize));
    let actual = F::from_bytes(&actual);

    let epsilon = epsilon.max(F::EPSILON.to_f32().unwrap());

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let allowed_error = (epsilon * e.to_f32().unwrap()).max(epsilon);

        // account for lower precision at higher values
        if print_instead_of_compare {
            println!("{:?}: {:?}, {:?}", i, a, e);
        } else {
            let actual_nan = f32::is_nan(a.to_f32().unwrap());
            let expected_nan = f32::is_nan(e.to_f32().unwrap());

            if actual_nan != expected_nan {
                if expected_nan {
                    return Err(format!("Expected NaN, got value={:?}", *a));
                } else {
                    return Err(format!("Expected value={:?}, got NaN", *e));
                }
            }

            let difference = f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap());

            if difference >= allowed_error {
                return Err(format!(
                    "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                    i, *a, *e, difference, epsilon
                ));
            }
        }
    }

    if print_instead_of_compare {
        Err("".to_string())
    } else {
        Ok(())
    }
}
