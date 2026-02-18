use fulaminas::backend::ndarray::NdArray;
use fulaminas::backend::Backend;
use fulaminas::engine::build;
use fulaminas::engine::tensor::Tensor;

#[test]
fn test_sigmoid() {
    let x_data = NdArray::from_vec(vec![0.0], &[1]);
    let x = Tensor::<NdArray>::new_input(vec![1]);
    let y = x.clone().sigmoid();
    let grad = y.grad(&x);

    let y_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let grad_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let _a1 = Tensor::assign(&y_out, &y, 0);
    let _a2 = Tensor::assign(&grad_out, &grad, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data)]);

    let y_val = executor.get_node_data(y.id()).unwrap();
    assert!((NdArray::to_vec(y_val)[0] - 0.5).abs() < 1e-6);

    let grad_val = executor.get_node_data(grad.id()).unwrap();
    assert!((NdArray::to_vec(grad_val)[0] - 0.25).abs() < 1e-6);
}

#[test]
fn test_tanh() {
    let x_data = NdArray::from_vec(vec![0.0], &[1]);
    let x = Tensor::<NdArray>::new_input(vec![1]);
    let y = x.clone().tanh();
    let grad = y.grad(&x);

    let y_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let grad_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let _a1 = Tensor::assign(&y_out, &y, 0);
    let _a2 = Tensor::assign(&grad_out, &grad, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data)]);

    let y_val = executor.get_node_data(y.id()).unwrap();
    assert!((NdArray::to_vec(y_val)[0] - 0.0).abs() < 1e-6);

    let grad_val = executor.get_node_data(grad.id()).unwrap();
    assert!((NdArray::to_vec(grad_val)[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_relu() {
    let x_data = NdArray::from_vec(vec![-1.0, 1.0], &[2]);
    let x = Tensor::<NdArray>::new_input(vec![2]);
    let y = x.clone().relu();
    let grad = y.grad(&x);

    let y_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[2]));
    let grad_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[2]));
    let _a1 = Tensor::assign(&y_out, &y, 0);
    let _a2 = Tensor::assign(&grad_out, &grad, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data)]);

    let y_val = executor.get_node_data(y.id()).unwrap();
    let y_vec = NdArray::to_vec(y_val);
    assert!((y_vec[0] - 0.0).abs() < 1e-6);
    assert!((y_vec[1] - 1.0).abs() < 1e-6);

    let grad_val = executor.get_node_data(grad.id()).unwrap();
    let grad_vec = NdArray::to_vec(grad_val);
    assert!((grad_vec[0] - 0.0).abs() < 1e-6);
    assert!((grad_vec[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_log_exp() {
    let x_data = NdArray::from_vec(vec![1.0], &[1]);
    let x = Tensor::<NdArray>::new_input(vec![1]);
    let y = x.clone().exp().log(); // Should be identity x
    let grad = y.grad(&x);

    let y_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let grad_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let _a1 = Tensor::assign(&y_out, &y, 0);
    let _a2 = Tensor::assign(&grad_out, &grad, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data)]);

    let y_val = executor.get_node_data(y.id()).unwrap();
    assert!((NdArray::to_vec(y_val)[0] - 1.0).abs() < 1e-6);

    let grad_val = executor.get_node_data(grad.id()).unwrap();
    assert!((NdArray::to_vec(grad_val)[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_powi() {
    let x_data = NdArray::from_vec(vec![2.0], &[1]);
    let x = Tensor::<NdArray>::new_input(vec![1]);
    let y = x.clone().powi(3);
    let grad = y.grad(&x);

    let y_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let grad_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[1]));
    let _a1 = Tensor::assign(&y_out, &y, 0);
    let _a2 = Tensor::assign(&grad_out, &grad, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data)]);

    let y_val = executor.get_node_data(y.id()).unwrap();
    assert!((NdArray::to_vec(y_val)[0] - 8.0).abs() < 1e-6);

    let grad_val = executor.get_node_data(grad.id()).unwrap();
    assert!((NdArray::to_vec(grad_val)[0] - 12.0).abs() < 1e-6);
}

#[test]
fn test_softmax() {
    // Softmax along axis 0
    let x_data = NdArray::from_vec(vec![1.0, 2.0], &[2]);
    let x = Tensor::<NdArray>::new_input(vec![2]);
    let y = x.clone().softmax(Some(0)); // or None for all
    let l = y.clone().sum(Some(0)); // Scalar 1.0
    let grad = l.grad(&x);

    let y_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[2]));
    let grad_out = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[2]));
    let _a1 = Tensor::assign(&y_out, &y, 0);
    let _a2 = Tensor::assign(&grad_out, &grad, 0);

    let mut executor = build::<NdArray>();
    executor.run(vec![(x, x_data)]);

    let y_val = executor.get_node_data(y.id()).unwrap();
    let y_vec = NdArray::to_vec(y_val);
    let sum = y_vec[0] + y_vec[1];
    assert!((sum - 1.0).abs() < 1e-6);

    let grad_val = executor.get_node_data(grad.id()).unwrap();
    let grad_vec = NdArray::to_vec(grad_val);

    // Gradients should be 0 because loss is constant 1.0
    assert!((grad_vec[0]).abs() < 1e-5);
    assert!((grad_vec[1]).abs() < 1e-5);
}
