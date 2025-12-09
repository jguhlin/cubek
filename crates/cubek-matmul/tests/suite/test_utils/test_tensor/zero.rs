use cubecl::{TestRuntime, prelude::*, std::tensor::TensorHandle, tensor_line_size_parallel};

#[cube(launch)]
fn zero_launch<T: Numeric>(tensor: &mut Tensor<Line<T>>, #[define(T)] _types: StorageType) {
    tensor[ABSOLUTE_POS] = Line::cast_from(0);
}

pub fn new_zeroed(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    let num_elems = shape.iter().product();

    let line_size = tensor_line_size_parallel(
        TestRuntime::supported_line_sizes().iter().copied(),
        &[num_elems],
        &[1],
        0,
    );

    let num_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::default();
    let cube_count = num_units as u32 / cube_dim.num_elems();

    let out =
        TensorHandle::new_contiguous(shape.clone(), client.empty(dtype.size() * num_elems), dtype);

    zero_launch::launch::<TestRuntime>(
        client,
        CubeCount::Static(cube_count, 1, 1),
        cube_dim,
        unsafe {
            TensorArg::from_raw_parts_and_size(
                &out.handle,
                &out.strides,
                &out.shape,
                line_size,
                dtype.size(),
            )
        },
        dtype,
    )
    .unwrap();

    out
}
