use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;
use cubecl::{TestRuntime, prelude::*};
use cubek_matmul::components::{MatmulIdent, MatmulProblem, MatrixLayout};
use cubek_matmul::tune_key::MatmulElemType;

use crate::suite::test_utils::test_tensor::cast::new_casted;

/// Returns tensor filled with zeros
pub(crate) fn output_test_tensor(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    dtype: MatmulElemType,
) -> TensorHandle<TestRuntime> {
    let tensor_shape = problem.shape(MatmulIdent::Out);
    TensorHandle::zeros(client, tensor_shape, *dtype)
}

/// Returns randomly generated input tensor to use in test along with a vec filled with the
/// same data as the handle
pub(crate) fn input_test_tensor(
    client: &ComputeClient<TestRuntime>,
    dtype: MatmulElemType,
    seed: u64,
    layout: MatrixLayout,
    tensor_shape: Vec<usize>,
) -> (TensorHandle<TestRuntime>, Vec<f32>) {
    // Create buffer with random elements that will be used in test
    cubek_random::seed(seed);
    let dtype = dtype.dtype;
    // Squash all dims because it's random and does not matter, this will make easier to reason for row/col major
    let shape_for_random = vec![tensor_shape.iter().product()];
    let tensor_handle = TensorHandle::empty(client, shape_for_random, dtype);

    cubek_random::random_uniform(
        &client,
        f32::from_int(-1),
        f32::from_int(1),
        tensor_handle.as_ref(),
        dtype,
    )
    .unwrap();

    // Obtain the data in f32 for comparison
    let data_handle = new_casted(client, &tensor_handle);
    let data_f32 =
        f32::from_bytes(&client.read_one_tensor(data_handle.as_copy_descriptor())).to_owned();

    match layout {
        MatrixLayout::RowMajor => (
            TensorHandle::new_contiguous(tensor_shape, tensor_handle.handle, tensor_handle.dtype),
            data_f32,
        ),
        MatrixLayout::ColMajor => {
            let batch = tensor_shape[..tensor_shape.len() - 2].iter().product();
            let rows = tensor_shape[tensor_shape.len() - 2];
            let cols = tensor_shape[tensor_shape.len() - 1];

            let mut strides = Vec::with_capacity(tensor_shape.len());
            let mut current = 1;
            tensor_shape.iter().enumerate().rev().for_each(|(_, val)| {
                strides.push(current);
                current *= val;
            });
            strides.reverse();
            strides.swap(tensor_shape.len() - 1, tensor_shape.len() - 2);

            let col_major_data = transpose(&data_f32, batch, rows, cols);

            (
                TensorHandle::new(
                    tensor_handle.handle,
                    tensor_shape,
                    strides,
                    tensor_handle.dtype,
                ),
                col_major_data,
            )
        }
    }
}

fn transpose(array: &[f32], batches: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![array[0]; array.len()];
    for b in 0..batches {
        for i in 0..rows {
            for j in 0..cols {
                result[(b * rows * cols) + j * rows + i] = array[(b * rows * cols) + i * cols + j];
            }
        }
    }
    result
}
