use std::fs::File;
use std::io::Write;

use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::engine::build;
use fulaminas::engine::layer::linear::InitStrategy;
use fulaminas::engine::layer::linear::Linear;
use fulaminas::engine::loss::CrossEntropyLoss;
use fulaminas::engine::optimizer::{Optimizer, SGD};
use fulaminas::engine::shape::{Rank0, Rank1, Rank2};
use fulaminas::engine::tensor::Tensor;

// XOR Learning with Static Shapes
fn main() {
    // XOR inputs
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

    // Static tensors: Batch=1, In=2, Hidden=5, Out=2
    let x = Tensor::<NdArray, Rank2<1, 2>>::new_input();
    let target = Tensor::<NdArray, Rank2<1, 2>>::new_input();

    // Model
    // Linear::<Backend, In, Out>
    let h1 = Linear::<NdArray, 2, 5>::new(InitStrategy::XavierNormal).forward(x.clone());
    let h2 = h1.tanh();
    let y = Linear::<NdArray, 5, 2>::new(InitStrategy::XavierNormal).forward(h2);

    // Loss
    // Returns Rank2<1, 2> element-wise
    let elem_loss = CrossEntropyLoss::new().forward(y, target.clone());

    // Reduce over classes (axis 1) -> Rank1<1>
    let loss_per_sample = elem_loss.sum_as::<Rank1<1>>(Some(1));

    // Reduce to scalar (Rank0)
    let loss = loss_per_sample.sum_as::<Rank0>(None);

    let mut optimizer = SGD::new(0.1);
    optimizer.step(&loss);

    let mut executor = build::<NdArray>();
    let dot = executor.to_dot();
    // println!("{}", dot);

    let mut file = File::create("graph.dot").unwrap();
    file.write_all(dot.as_bytes()).unwrap();

    for j in 0..10000 {
        let i = j % 4;
        executor.run(vec![
            (x.id(), NdArray::from_vec(x_data[i].clone(), &[1, 2])),
            (target.id(), NdArray::from_vec(y_data[i].clone(), &[1, 2])),
        ]);

        if j % 1000 == 0 {
            // Loss is Rank0
            let loss_val_tensor = executor.get_node_data(loss.id()).unwrap();
            let val = NdArray::to_vec(loss_val_tensor)[0]; // Rank0 data is usually 1 element
            println!("Iter: {}, Loss: {}", j, val);
        }
    }
}
