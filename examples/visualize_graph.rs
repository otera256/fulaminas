use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::engine::{build, shape::Dynamic, tensor::Tensor};
use std::fs::File;
use std::io::Write;

type DTensor = Tensor<NdArray, Dynamic>;

fn main() {
    // y = 3 * x^2
    // dy/dx = 6 * x
    let x = DTensor::new_input_dynamic(vec![1]);
    let c3 = DTensor::new_const(NdArray::from_vec(vec![3.0], &[1]));
    let y = c3 * x.clone() * x.clone();
    let grad_x = y.grad(&x);

    // Assign to output to enforce calculation
    let output = DTensor::new_parameter(NdArray::zeros(&[1]));
    let _assign = DTensor::assign(&output, &grad_x, 0);

    let executor = build::<NdArray>();

    let dot = executor.to_dot();
    println!("{}", dot);

    let mut file = File::create("graph.dot").unwrap();
    file.write_all(dot.as_bytes()).unwrap();
    println!("Graph written to graph.dot");
}
