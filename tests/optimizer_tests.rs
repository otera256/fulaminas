use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::engine::build;
use fulaminas::engine::optimizer::{Adam, Optimizer, SGD};
use fulaminas::engine::shape::Dynamic;
use fulaminas::engine::tensor::Tensor;

type DTensor = Tensor<NdArray, Dynamic>;

#[test]
fn test_sgd_convergence() {
    // Minimize y = x^2, starting at x=2. Min at x=0.
    // L = x^2. dL/dx = 2x.
    // x_new = x - lr * 2x.
    // If lr = 0.1, x_new = x - 0.2x = 0.8x.
    // After n steps, x -> 0.

    let x_data = NdArray::from_vec(vec![2.0], &[1]);

    // Create Parameter x
    let x = DTensor::new_parameter(x_data.clone());

    // Loss
    let loss = x.clone() * x.clone();

    // Optimizer
    let mut optimizer = SGD::new(0.1);
    optimizer.update_param(
        &x,
        &loss.clone().sum_as::<fulaminas::engine::shape::Rank0>(None),
    ); // Builds update graph

    let mut executor = build::<NdArray>();

    // Run multiple steps
    for _ in 0..10 {
        executor.run(vec![]);
    }

    let x_val = executor.get_node_data(x.id()).unwrap();
    let val = NdArray::to_vec(x_val)[0];

    // Should be close to 0
    assert!(val.abs() < 0.3, "SGD did not converge, got {}", val);
    assert!(val < 2.0, "SGD did not decrease");
}

#[test]
fn test_adam_convergence() {
    let x_data = NdArray::from_vec(vec![2.0], &[1]);
    let x = DTensor::new_parameter(x_data.clone());

    let loss = x.clone() * x.clone();

    let mut optimizer = Adam::new(0.1);
    optimizer.update_param(
        &x,
        &loss.clone().sum_as::<fulaminas::engine::shape::Rank0>(None),
    );

    let mut executor = build::<NdArray>();

    for _ in 0..50 {
        executor.run(vec![]);
    }

    let x_val = executor.get_node_data(x.id()).unwrap();
    let val = NdArray::to_vec(x_val)[0];

    assert!(val.abs() < 0.5, "Adam did not converge, got {}", val);
    assert!(val < 2.0, "Adam did not decrease");
}
