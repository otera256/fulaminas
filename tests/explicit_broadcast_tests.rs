use fulaminas::backend::{Backend, ndarray::NdArray};
use fulaminas::engine::{
    build,
    optimizer::{Optimizer, SGD},
    shape::Dynamic,
    tensor::Tensor,
};

#[test]
fn test_explicit_broadcast_add() {
    type B = NdArray;
    type DTensor = Tensor<B, Dynamic>;

    // a: [2, 3], b: [1, 3]
    let a_data = B::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b_data = B::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]);

    let a = DTensor::new_parameter(a_data);
    let b = DTensor::new_parameter(b_data);

    // c = a + b (explicit broadcast expected for b)
    let c = a + b;

    // Create an output node to hold the result
    let output = DTensor::new_input_dynamic(vec![2, 3]);
    let _ = Tensor::assign(&output, &c, 0);

    let mut executor = build::<B>();

    executor.step_inference(vec![]);

    // We expect 'output' to hold the result of c
    let result = executor.get_node_data(output.id()).unwrap();
    let result_vec = B::to_vec(&result);

    assert_eq!(result_vec, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_explicit_broadcast_grad() {
    type B = NdArray;
    type DTensor = Tensor<B, Dynamic>;

    let a_data = B::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b_data = B::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]);

    let a = DTensor::new_parameter(a_data);
    let b = DTensor::new_parameter(b_data);
    let b_id = b.id();

    let c = a.clone() * b.clone();
    let loss = c.sum(None); // Sum all

    // Optimizer step creates assignment nodes which are roots.
    let mut optimizer = SGD::new(1.0);
    optimizer.update_param(
        &a,
        &loss.clone().sum_as::<fulaminas::engine::shape::Rank0>(None),
    );
    optimizer.update_param(
        &b,
        &loss.clone().sum_as::<fulaminas::engine::shape::Rank0>(None),
    );

    let mut executor = build::<B>();
    executor.step_train(vec![]);

    // Get updated b
    let updated_b = executor.get_node_data(b_id).unwrap();
    let updated_b_vec = B::to_vec(&updated_b);

    // Expected gradient for b is [5, 7, 9] (sum of a along axis 0)
    // b_new = b - 1.0 * grad
    // b = [10, 20, 30]
    // b_new = [5, 13, 21]
    assert_eq!(updated_b_vec, vec![5.0, 13.0, 21.0]);
}
