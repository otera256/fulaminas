use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::engine::build;
use fulaminas::engine::layer::{
    Layer,
    linear::{InitStrategy, Linear},
};
use fulaminas::engine::shape::{Rank1, Rank2};
use fulaminas::engine::tensor::Tensor;

#[test]
fn test_linear_forward_shape() {
    const IN_FEATURES: usize = 3;
    const OUT_FEATURES: usize = 2;
    const BATCH: usize = 4;

    let layer = Linear::<NdArray, IN_FEATURES, OUT_FEATURES>::new(InitStrategy::XavierNormal);

    // Batch size 4, input features 3 -> [4, 3]
    let x_data = NdArray::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[BATCH, IN_FEATURES],
    );
    // Static input
    let x = Tensor::<NdArray, Rank2<BATCH, IN_FEATURES>>::new_input();

    let y = layer.forward(x.clone());

    // Result must be assigned to be computed by executor
    let y_out = Tensor::<NdArray, Rank2<BATCH, OUT_FEATURES>>::new_parameter(NdArray::zeros(&[
        BATCH,
        OUT_FEATURES,
    ]));
    let _ = Tensor::assign(&y_out, &y, 0);

    // Output should be [4, 2]
    // We need to force execution to check shape/result
    // The result y is static Tensor<NdArray, Rank2<4, 2>>.

    let mut executor = build::<NdArray>();
    executor.run(vec![(x.id(), x_data)]);

    let y_val = executor.get_node_data(y.id()).unwrap();
    let shape = NdArray::shape(&y_val);
    assert_eq!(shape, vec![BATCH, OUT_FEATURES]);
}

#[test]
fn test_linear_initialization_stats() {
    // Large enough to check statistics
    const IN_FEATURES: usize = 100;
    const OUT_FEATURES: usize = 100;

    // HeNormal: std = sqrt(2/in) = sqrt(0.02) ~= 0.1414
    let layer = Linear::<NdArray, IN_FEATURES, OUT_FEATURES>::new(InitStrategy::HeNormal);
    let w = layer.w.clone(); // weight (Static Rank2)
    let _b = layer.b.clone(); // bias

    // We can't easily get data from Tensor without running a graph.
    // But parameters created with new_parameter (inside Linear::new) have data.
    // We can use a dummy graph to retrieve it.

    let mut executor = build::<NdArray>();
    // No inputs needed just to read params
    executor.run(vec![]);

    let w_val = executor.get_node_data(w.id()).unwrap();
    let w_vec = NdArray::to_vec(w_val);

    let mean: f32 = w_vec.iter().sum::<f32>() / w_vec.len() as f32;
    let variance: f32 = w_vec.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / w_vec.len() as f32;
    let std = variance.sqrt();

    // Mean should be close to 0
    assert!(
        mean.abs() < 0.05,
        "Mean {} should be likely close to 0",
        mean
    );

    // Std should be close to 0.1414
    assert!(
        (std - 0.1414).abs() < 0.05,
        "Std {} should be likely close to 0.1414",
        std
    );
}

#[test]
fn test_linear_backward() {
    const IN_FEATURES: usize = 2;
    const OUT_FEATURES: usize = 1;
    let layer = Linear::<NdArray, IN_FEATURES, OUT_FEATURES>::new(InitStrategy::XavierUniform);

    let x_data = NdArray::from_vec(vec![1.0, 2.0], &[1, 2]);
    let x = Tensor::<NdArray, Rank2<1, 2>>::new_input();

    let y = layer.forward(x.clone());

    // Loss = y^2
    let loss = y.clone() * y.clone();

    // loss is Rank2<1, 1>.
    // To get gradients, we call loss.grad(&w).
    // But w is inside layer. `layer` struct has `w` field but private?
    // We can use `layer.parameters()` to get dynamic handles.
    // `Tensor::grad` works on generic `Tensor`.
    // But `loss` is static Tensor. `params` are dynamic tensors (returned by parameters()).
    // If we want grad w.r.t static parameter, we need access to it.
    // `Linear` fields are private.
    // BUT `Tensor::grad` takes `&T` where `T` is `Tensor`.
    // We can cast `loss` to dynamic, or cast params to static?
    // Actually, `Linear` struct definition: `w: Tensor<B, Rank2<I, O>>`.
    // If fields are private, we can't access `layer.w` in tests unless we make them pub or use accessors.
    // `parameters()` returns clones/conversions to dynamic.
    // If we use `loss.grad(&param_dyn)`, does it work?
    // `loss` is `Tensor<B, S>`. `param_dyn` is `Tensor<B, Dynamic>`.
    // `id()` matches.
    // `grad` implementation:
    // `pub fn grad<S2: Shape>(&self, target: &Tensor<B, S2>) -> Tensor<B, S2>`?
    // `Tensor::grad` returns `Self` (gradient w.r.t target has same shape as target).
    // So `loss.grad(&param)` returns tensor of same shape/type as `param`.
    // If `param` is dynamic, `grad` returns dynamic. This is fine.

    let w = layer.w.clone();
    let b = layer.b.clone();

    let grad_w = loss
        .clone()
        .sum_as::<fulaminas::engine::shape::Rank0>(None)
        .grad(&w);
    let grad_b = loss
        .clone()
        .sum_as::<fulaminas::engine::shape::Rank0>(None)
        .grad(&b);

    // Assign grads to outputs
    let grad_w_out =
        Tensor::<NdArray, Rank2<IN_FEATURES, OUT_FEATURES>>::new_parameter(NdArray::zeros(&[
            IN_FEATURES,
            OUT_FEATURES,
        ]));
    let grad_b_out =
        Tensor::<NdArray, Rank1<OUT_FEATURES>>::new_parameter(NdArray::zeros(&[OUT_FEATURES]));

    // We need to match shapes. grad_w is dynamic from loss.grad.
    // We need to cast or assign dynamic to static?
    // Tensor::assign<S> expects same S.
    // w is dynamic parameters(). grad is dynamic.
    // grad_w_out is static.
    // We can use grad_w_out.to_dynamic() as target?
    // And assign.
    let _ = Tensor::assign(&grad_w_out, &grad_w, 0);
    let _ = Tensor::assign(&grad_b_out, &grad_b, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x.id(), x_data)]);

    let gw_val = executor.get_node_data(grad_w.id()).unwrap();
    let gb_val = executor.get_node_data(grad_b.id()).unwrap();

    let gw_vec = NdArray::to_vec(gw_val);
    let gb_vec = NdArray::to_vec(gb_val);

    assert!(gw_vec.iter().any(|&v| v.abs() > 1e-6));
    assert!(gb_vec.iter().any(|&v| v.abs() > 1e-6));
}
