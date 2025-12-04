macro_rules! suite_dtype {
    ($module:expr) => {
        mod f32 {
            type TestDType = f32;

            include!($module);
        }

        mod f16 {
            type TestDType = half::f16;

            include!($module);
        }
    };
}

mod normal {
    suite_dtype!("normal.rs");
}

mod bernoulli {
    suite_dtype!("bernoulli.rs");
}

mod interval {
    include!("interval.rs");
}

mod uniform {
    type TestDType = f32;

    include!("uniform.rs");
}
