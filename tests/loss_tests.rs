use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::engine::build;
use fulaminas::engine::loss::{CrossEntropyLoss, MSELoss};
use fulaminas::engine::shape::Dynamic;
use fulaminas::engine::tensor::Tensor;

type DTensor = Tensor<NdArray, Dynamic>;

#[test]
fn test_mse_loss() {
    let loss_fn = MSELoss::new();

    // Pred: [1, 2, 3]
    // Target: [1, 2, 5]
    // Diff: [0, 0, -2]
    // Sq diff: [0, 0, 4]
    // Mean: 4/3 ~= 1.333

    let pred_data = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let target_data = NdArray::from_vec(vec![1.0, 2.0, 5.0], &[3]);

    let pred = DTensor::new_input_dynamic(vec![3]);
    let target = DTensor::new_input_dynamic(vec![3]);

    let elem_loss = loss_fn.forward::<NdArray, Dynamic>(pred.clone(), target.clone());
    let loss = elem_loss.sum_as::<Dynamic>(None); // Sum all

    // We want Mean. Sum is 4. Mean is 4/3.
    // The previous implementation utilized Mean reduction internally.
    // Here we just test that the graph computes correctly.
    // Let's divide by 3 manually or just check Sum = 4.

    let loss_out = DTensor::new_parameter(NdArray::zeros(&[]));
    let _assign = DTensor::assign(&loss_out, &loss, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(pred.id(), pred_data), (target.id(), target_data)]);

    let loss_val = executor.get_node_data(loss.id()).unwrap();
    // Sum is 4.0
    assert!((NdArray::to_vec(loss_val)[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_mse_grad() {
    let loss_fn = MSELoss::new();

    // y = x
    // target = 0
    // L = mean((x)^2) = x^2 (since dim=1)
    // dL/dx = 2x
    // x = 2 -> grad = 4

    let x_data = NdArray::from_vec(vec![2.0], &[1]);
    let t_data = NdArray::from_vec(vec![0.0], &[1]);

    let x = DTensor::new_input_dynamic(vec![1]);
    let t = DTensor::new_input_dynamic(vec![1]);

    let elem_loss = loss_fn.forward::<NdArray, Dynamic>(x.clone(), t.clone());
    // loss = (x-t)^2. Gradient w.r.t x is 2(x-t).
    // We don't reduce for this test?
    // "L = mean((x)^2)".
    // If we want mean, we should reduce.
    // Previous test expected 4.0. which is 2*x (x=2).
    // This implies d/dx (x^2) = 2x.
    // If we don't reduce, L is a tensor. grad will be computed element-wise?
    // Tensor::grad expects target to be scalar usually for backward?
    // "grad" method in Tensor creates a Grad node.
    // It works for any shape (backprop from L to x).
    // If L is [1], grad is [1].
    // If x=[1], t=[1], elem_loss=[1]. output is scalar-like.
    let loss = elem_loss; // It is already [1] because inputs are [1].
    let grad = loss.grad(&x);

    let grad_out = DTensor::new_parameter(NdArray::zeros(&[1]));
    let _assign = DTensor::assign(&grad_out, &grad, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x.id(), x_data), (t.id(), t_data)]);

    let grad_val = executor.get_node_data(grad.id()).unwrap();
    // MSE = sum(diff^2) * (1/N). N=1.
    // d/dx (x-0)^2 = 2x. 2*2 = 4.
    assert!((NdArray::to_vec(grad_val)[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_loss() {
    let loss_fn = CrossEntropyLoss::new();

    // Batch=1, Classes=2
    // Logits: [0.0, 0.0] -> Softmax: [0.5, 0.5]
    // Target: [1.0, 0.0] (Class 0)
    // Loss: - (1 * log(0.5) + 0 * log(0.5)) = -log(0.5) = log(2) ~= 0.6931

    let logits_data = NdArray::from_vec(vec![0.0, 0.0], &[1, 2]);
    let target_data = NdArray::from_vec(vec![1.0, 0.0], &[1, 2]);

    let logits = DTensor::new_input_dynamic(vec![1, 2]);
    let target = DTensor::new_input_dynamic(vec![1, 2]);

    let elem_loss = loss_fn.forward::<NdArray, Dynamic>(logits.clone(), target.clone());
    // elem_loss is [1, 2].
    // We want to sum over classes to get total loss per sample (which is 0.6931).
    let loss = elem_loss.sum_as::<Dynamic>(Some(1));

    let loss_out = DTensor::new_parameter(NdArray::zeros(&[1]));
    let _assign = DTensor::assign(&loss_out, &loss, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(logits.id(), logits_data), (target.id(), target_data)]);

    let loss_val = executor.get_node_data(loss.id()).unwrap();
    assert!((NdArray::to_vec(loss_val)[0] - 0.693147).abs() < 1e-5);
}
