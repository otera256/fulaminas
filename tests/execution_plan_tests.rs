use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::engine::build;
use fulaminas::engine::optimizer::{Optimizer, SGD};
use fulaminas::engine::shape::Rank0;
use fulaminas::engine::tensor::Tensor;

#[test]
fn test_execution_plan_separation() {
    // 1. Create a parameter p initialized to 10.0
    let p = Tensor::<NdArray, Rank0>::new_parameter(NdArray::from_vec(vec![10.0], &[]));

    // 2. Define Loss = p (minimize p to 0)
    // We need a dummy input just to have something to feed? No, can use p directly.
    let loss = p.clone();

    // 3. Create Optimizer
    let mut optimizer = SGD::new(1.0); // Learning rate 1.0
    // step adds Assign nodes: p = p - lr * grad
    // grad of p w.r.t p is 1.0.
    // So new_p should be 10.0 - 1.0 * 1.0 = 9.0 after training step.
    optimizer.step(&loss);

    let mut executor = build::<NdArray>();

    // 4. Initial check
    let p_val_0 = executor.get_node_data(p.id()).unwrap();
    assert_eq!(NdArray::to_vec(p_val_0)[0], 10.0);

    // 5. Run Inference Step
    // Should NOT update p.
    executor.step_inference(vec![]);

    // Check p value again
    let p_val_inf = executor.get_node_data(p.id()).unwrap();
    assert_eq!(
        NdArray::to_vec(p_val_inf)[0],
        10.0,
        "Inference step should not update parameters"
    );

    // 6. Run Training Step
    // Should update p to 9.0.
    executor.step_train(vec![]);

    let p_val_train = executor.get_node_data(p.id()).unwrap();
    assert_eq!(
        NdArray::to_vec(p_val_train)[0],
        9.0,
        "Training step should update parameters"
    );
}
