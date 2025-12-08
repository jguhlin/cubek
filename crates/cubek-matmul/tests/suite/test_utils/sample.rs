use cubecl::{Runtime, client::ComputeClient, flex32, prelude::*, std::tensor::TensorHandle, tf32};

pub trait Sample: Sized + CubePrimitive {
    fn sample<R: Runtime>(client: &ComputeClient<R>, shape: &[usize], seed: u64)
    -> TensorHandle<R>;
}

macro_rules! sample_float {
    ($($t:ty),*) => {
        $(
            impl Sample for $t
            {
                fn sample<R: Runtime>(client: &ComputeClient<R>, shape: &[usize], seed: u64) -> TensorHandle<R> {
                    cubek_random::seed(seed);
                    let dtype = Self::as_type_native_unchecked();
                    let output = TensorHandle::empty(client, shape.to_vec(), dtype);

                    cubek_random::random_uniform(&client, f32::from_int(-1), f32::from_int(1), output.as_ref(), dtype).unwrap();

                    output
                }
            }
        )*
    };
}

sample_float!(half::f16);
sample_float!(half::bf16);
sample_float!(f32);
sample_float!(f64);
sample_float!(u8);

impl Sample for flex32 {
    fn sample<R: Runtime>(
        client: &ComputeClient<R>,
        shape: &[usize],
        seed: u64,
    ) -> TensorHandle<R> {
        cubek_random::seed(seed);
        let dtype = f32::as_type_native_unchecked();
        let output = TensorHandle::empty(client, shape.to_vec(), dtype);

        cubek_random::random_uniform(
            client,
            f32::from_int(-1),
            f32::from_int(1),
            output.as_ref(),
            dtype,
        )
        .unwrap();

        output
    }
}

impl Sample for tf32 {
    fn sample<R: Runtime>(
        client: &ComputeClient<R>,
        shape: &[usize],
        seed: u64,
    ) -> TensorHandle<R> {
        cubek_random::seed(seed);
        let dtype = f32::as_type_native_unchecked();
        let output = TensorHandle::empty(client, shape.to_vec(), dtype);

        cubek_random::random_uniform(
            client,
            f32::from_int(-1),
            f32::from_int(1),
            output.as_ref(),
            dtype,
        )
        .unwrap();

        output
    }
}
