use std::fs::File;
use std::io::Write;

use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::engine::build;
use fulaminas::engine::layer::{Layer, Linear};
use fulaminas::engine::loss::CrossEntropyLoss;
use fulaminas::engine::optimizer::{Optimizer, SGD};
use fulaminas::engine::tensor::Tensor;

// XORゲートの学習
fn main() {
    // XORゲートの入力
    let x_data = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    // One-hot encoding for 2 classes: 0 -> [1, 0], 1 -> [0, 1]
    let y_data = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];
    // XORゲートの出力 (One-hot)
    let labels_placeholder = Tensor::<NdArray>::new_input(vec![1, 2]);

    // モデルの構築
    let x = Tensor::<NdArray>::new_input(vec![1, 2]);
    let h1 =
        Linear::new(2, 5, fulaminas::engine::layer::InitStrategy::XavierNormal).forward(x.clone());
    let h2 = h1.tanh();
    let h3 = Linear::new(5, 2, fulaminas::engine::layer::InitStrategy::XavierNormal).forward(h2);
    // CrossEntropyLoss computes Softmax internally
    let y = h3;

    let loss = CrossEntropyLoss::new().forward(y, labels_placeholder.clone());
    let mut optimizer = SGD::new(0.1);
    optimizer.step(&loss);
    let mut executor = build::<NdArray>();
    let dot = executor.to_dot();
    println!("{}", dot);

    let mut file = File::create("graph.dot").unwrap();
    file.write_all(dot.as_bytes()).unwrap();
    println!("Graph written to graph.dot");
    for j in 0..10000 {
        let i = j % 4;
        executor.run(vec![
            (x.clone(), NdArray::from_vec(x_data[i].clone(), &[1, 2])),
            (
                labels_placeholder.clone(),
                NdArray::from_vec(y_data[i].clone(), &[1, 2]),
            ),
        ]);
        if j % 1000 == 0 {
            // Loss is a scalar tensor
            let loss_val = executor.get_node_data(loss.id()).unwrap();
            println!(
                "Loss: {}",
                loss_val
                    .get(ndarray::IxDyn(&[]))
                    .unwrap_or(&loss_val.first().unwrap())
            );
        }
    }
}
