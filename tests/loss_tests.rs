use fulaminas::backend::ndarray::NdArray;
use fulaminas::backend::Backend;
use fulaminas::engine::build;
use fulaminas::engine::loss::{CrossEntropyLoss, MSELoss};
use fulaminas::engine::tensor::Tensor;

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

    let pred = Tensor::<NdArray>::new_input(vec![3]);
    let target = Tensor::<NdArray>::new_input(vec![3]);

    let loss = loss_fn.forward(pred.clone(), target.clone());

    let loss_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let _assign = Tensor::assign(&loss_out, &loss, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(pred, pred_data), (target, target_data)]);

    let loss_val = executor.get_node_data(loss.id()).unwrap();
    assert!((NdArray::to_vec(loss_val)[0] - 1.333333).abs() < 1e-5);
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

    let x = Tensor::<NdArray>::new_input(vec![1]);
    let t = Tensor::<NdArray>::new_input(vec![1]);

    let loss = loss_fn.forward(x.clone(), t.clone());
    let grad = loss.grad(&x);

    let grad_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let _assign = Tensor::assign(&grad_out, &grad, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data), (t, t_data)]);

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

    let logits = Tensor::<NdArray>::new_input(vec![1, 2]);
    let target = Tensor::<NdArray>::new_input(vec![1, 2]);

    let loss = loss_fn.forward(logits.clone(), target.clone());

    let loss_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let _assign = Tensor::assign(&loss_out, &loss, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(logits, logits_data), (target, target_data)]);

    let loss_val = executor.get_node_data(loss.id()).unwrap();
    assert!((NdArray::to_vec(loss_val)[0] - 0.693147).abs() < 1e-5);
}
