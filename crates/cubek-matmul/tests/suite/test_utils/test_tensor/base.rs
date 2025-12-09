use cubecl::std::tensor::TensorHandle;
use cubecl::{TestRuntime, prelude::*};
use cubecl::{
    client::ComputeClient,
    server::{Allocation, AllocationDescriptor},
};
use cubek_matmul::components::{MatmulIdent, MatmulProblem, MatrixLayout};
use cubek_matmul::tune_key::MatmulElemType;

use crate::suite::test_utils::test_tensor::cast::new_casted;
use crate::suite::test_utils::test_tensor::zero::new_zeroed;

/// Returns tensor filled with zeros
pub(crate) fn output_test_tensor(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    dtype: MatmulElemType,
) -> TensorHandle<TestRuntime> {
    let tensor_shape = problem.shape(MatmulIdent::Out);
    new_zeroed(client, tensor_shape, *dtype)
}

/// Returns randomly generated input tensor to use in test along with a vec filled with the
/// same data as the handle
pub(crate) fn input_test_tensor(
    client: &ComputeClient<TestRuntime>,
    dtype: MatmulElemType,
    seed: u64,
    layout: MatrixLayout,
    mut tensor_shape: Vec<usize>,
) -> (TensorHandle<TestRuntime>, Vec<f32>) {
    // Create buffer with random elements that will be used in test
    cubek_random::seed(seed);
    let dtype = dtype.dtype;
    let tensor_handle = TensorHandle::empty(client, tensor_shape.to_vec(), dtype);

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
    let data = client.read_one_tensor(data_handle.as_copy_descriptor());
    let data = f32::from_bytes(&data);
    let original_data = data.to_owned();

    // If col major we will rewrite the buffer in col major
    let rank = tensor_shape.len();
    let data = match layout {
        MatrixLayout::RowMajor => original_data.clone(),
        MatrixLayout::ColMajor => {
            tensor_shape.swap(rank - 1, rank - 2);
            transpose(
                &original_data,
                tensor_shape
                    .iter()
                    .take(tensor_shape.len().saturating_sub(2))
                    .product(),
                tensor_shape[rank - 1],
                tensor_shape[rank - 2],
            )
        }
    };
    let descriptors = vec![(
        AllocationDescriptor::optimized(tensor_shape.as_slice(), dtype.size()),
        bytemuck::cast_slice(&data),
    )];

    let mut tensors = client.create_tensors_from_slices(descriptors);
    let Allocation {
        handle,
        mut strides,
    } = tensors.remove(0);

    if matches!(layout, MatrixLayout::ColMajor) {
        tensor_shape.swap(rank - 1, rank - 2);
        strides.swap(rank - 1, rank - 2);
    }

    let _offs = tensors.pop();

    (
        TensorHandle::new(handle, tensor_shape, strides, dtype),
        original_data,
    )
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
