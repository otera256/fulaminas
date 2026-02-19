use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::engine::build;
use fulaminas::engine::layer::{
    Layer,
    linear::{InitStrategy, Linear},
};
use fulaminas::engine::tensor::Tensor;

#[test]
fn test_linear_forward_shape() {
    let in_features = 3;
    let out_features = 2;
    let layer = Linear::<NdArray>::new(in_features, out_features, InitStrategy::XavierNormal);

    // Batch size 4, input features 3 -> [4, 3]
    let x_data = NdArray::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    );
    let x = Tensor::<NdArray>::new_input(vec![4, 3]);

    let y = layer.forward(x.clone());

    // Output should be [4, 2]
    // We need to force execution to check shape/result
    let y_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[4, 2]));
    let _assign = Tensor::assign(&y_out, &y, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data)]);

    let y_val = executor.get_node_data(y.id()).unwrap();
    let shape = NdArray::shape(&y_val);
    assert_eq!(shape, vec![4, 2]);
}

#[test]
fn test_linear_initialization_stats() {
    // Large enough to check statistics
    let in_features = 100;
    let out_features = 100;

    // HeNormal: std = sqrt(2/in) = sqrt(0.02) ~= 0.1414
    let layer = Linear::<NdArray>::new(in_features, out_features, InitStrategy::HeNormal);
    let params = layer.parameters();
    let w = &params[0]; // weight
    let _b = &params[1]; // bias

    // We can't easily get data from Tensor without running a graph,
    // but parameters created with new_parameter should have data immediately available
    // IF we could access it.
    // However, Tensor architecture might hide it in GraphBuilder.
    // Actually, `Tensor::new_parameter` creates a Node with data.
    // We can use a dummy graph to retrieve it.

    let w_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[in_features, out_features]));
    let _assign = Tensor::assign(&w_out, w, 0);

    let mut executor = build::<NdArray>();
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
    let in_features = 2;
    let out_features = 1;
    let layer = Linear::<NdArray>::new(in_features, out_features, InitStrategy::XavierUniform);

    let x_data = NdArray::from_vec(vec![1.0, 2.0], &[1, 2]);
    let x = Tensor::<NdArray>::new_input(vec![1, 2]);

    let y = layer.forward(x.clone());

    // Loss = y^2
    let loss = y.clone() * y.clone();

    let params = layer.parameters();
    let w = &params[0];
    let b = &params[1];

    let grad_w = loss.grad(w);
    let grad_b = loss.grad(b);

    let grad_w_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[in_features, out_features]));
    let grad_b_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[out_features]));

    let _a1 = Tensor::assign(&grad_w_out, &grad_w, 0);
    let _a2 = Tensor::assign(&grad_b_out, &grad_b, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data)]);

    let gw_val = executor.get_node_data(grad_w.id()).unwrap();
    let gb_val = executor.get_node_data(grad_b.id()).unwrap();

    // Just check they are not all zeros (unless initialized exactly to 0 which is unlikely for W)
    // For XavierUniform, W is not 0.
    // If x=[1, 2], y = xW + b.
    // L = y^2. dL/dW = 2y * x. dL/db = 2y.

    let gw_vec = NdArray::to_vec(gw_val);
    let gb_vec = NdArray::to_vec(gb_val);

    assert!(gw_vec.iter().any(|&v| v.abs() > 1e-6));
    assert!(gb_vec.iter().any(|&v| v.abs() > 1e-6));
}
