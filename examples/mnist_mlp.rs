// MNIST and FashionMNIST MLP with Static Shapes

use std::fs::File;
use std::io::Write;

use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::data::dataset::Dataset;
use fulaminas::data::loader::DataLoader;
use fulaminas::data::mnist::Mnist;
use fulaminas::engine::build;
use fulaminas::engine::layer::linear::{InitStrategy, Linear};
use fulaminas::engine::loss::CrossEntropyLoss;
use fulaminas::engine::metric::Accuracy;
use fulaminas::engine::optimizer::{Adam, Optimizer};
use fulaminas::engine::shape::{Rank0, Rank1, Rank2, Rank4};
use fulaminas::engine::tensor::Tensor;

// Define constraints for network
const BATCH_SIZE: usize = 256;
const IN_FEATURES: usize = 784;
const HIDDEN_1: usize = 512;
const HIDDEN_2: usize = 256;
const HIDDEN_3: usize = 128;
const HIDDEN_4: usize = 64;
const HIDDEN_5: usize = 32;
const CLASSES: usize = 10;

fn main() {
    let mnist = Mnist::new(
        "./data/mnist",
        true,
        false, // Don't download, use local
        fulaminas::data::mnist::MnistVariant::Mnist,
    )
    .unwrap();
    let mnist_loader = DataLoader::new(&mnist, BATCH_SIZE, true);
    let mnist_validation = Mnist::new(
        "./data/mnist",
        false,
        false,
        fulaminas::data::mnist::MnistVariant::Mnist,
    )
    .unwrap();
    let mnist_validation_loader = DataLoader::new(&mnist_validation, BATCH_SIZE, true);

    // Static Input Tensors
    // x: [Batch, 1, 28, 28] -> Rank4
    let x = Tensor::<NdArray, Rank4<BATCH_SIZE, 1, 28, 28>>::new_input();
    let target = Tensor::<NdArray, Rank2<BATCH_SIZE, CLASSES>>::new_input();

    // Flatten: [Batch, 1, 28, 28] -> [Batch, 784]
    // Use reshape with explicit static target type
    // Clone x because reshape consumes it, but we need x for feeding input
    let x_flat = x.clone().reshape::<Rank2<BATCH_SIZE, IN_FEATURES>>();

    // Layers (Static)
    // Linear::<Backend, Input, Output>
    let layer1 = Linear::<NdArray, IN_FEATURES, HIDDEN_1>::new(InitStrategy::HeNormal);
    let layer2 = Linear::<NdArray, HIDDEN_1, HIDDEN_2>::new(InitStrategy::HeNormal);
    let layer3 = Linear::<NdArray, HIDDEN_2, HIDDEN_3>::new(InitStrategy::HeNormal);
    let layer4 = Linear::<NdArray, HIDDEN_3, HIDDEN_4>::new(InitStrategy::HeNormal);
    let layer5 = Linear::<NdArray, HIDDEN_4, HIDDEN_5>::new(InitStrategy::HeNormal);
    let layer6 = Linear::<NdArray, HIDDEN_5, CLASSES>::new(InitStrategy::HeNormal);

    // Forward pass
    // Note: linear.forward(x) expects Tensor<B, Rank2<Batch, In>>
    let h1 = layer1.forward(x_flat).relu();
    let h2 = layer2.forward(h1).relu();
    let h3 = layer3.forward(h2).relu();
    let h4 = layer4.forward(h3).relu();
    let h5 = layer5.forward(h4).relu();
    let y = layer6.forward(h5);

    println!("Graph built with Static Shapes.");

    // Loss
    // Returns [Batch, Classes] (element-wise nll)
    let elem_loss = CrossEntropyLoss::new().forward(y.clone(), target.clone());

    // Reduce to [Batch] (sum over classes)
    // Rank2<B, C> -> Rank1<B>
    // We use sum_as with axis=1.
    let loss_per_sample = elem_loss.sum_as::<Rank1<BATCH_SIZE>>(Some(1));

    // Reduce to scalar (Mean)
    // sum_as::<Rank0> (Total loss)
    let total_loss = loss_per_sample.sum_as::<Rank0>(None);

    // Scale by 1/BATCH_SIZE
    let scale =
        Tensor::<NdArray, Rank0>::new_const(NdArray::from_vec(vec![1.0 / BATCH_SIZE as f32], &[]));

    let loss = total_loss * scale;

    // Accuracy
    let acc = Accuracy::forward(y, target.clone());

    // Force computation by assigning to dummy params
    // We can use new_parameter to create a holder.
    // Initial value doesn't matter, will be overwritten.
    let loss_container = Tensor::<NdArray, Rank0>::new_input();
    let acc_container = Tensor::<NdArray, Rank0>::new_input();

    // We depend on side-effect of assign being a root node
    let _ = Tensor::assign(&loss_container, &loss, 0);
    let _ = Tensor::assign(&acc_container, &acc, 0);

    let mut optimizer = Adam::new(0.001);
    // optimizer.step requires Tensor<B, S>
    optimizer.step(&loss);

    let mut executor = build::<NdArray>();

    let dot = executor.to_dot();

    let mut file = File::create("graph.dot").unwrap();
    file.write_all(dot.as_bytes()).unwrap();
    println!("Graph written to graph.dot");

    for epoch in 0..10 {
        for (i, (input_val_4d, target_val, _mask_val)) in mnist_loader
            .iter()
            .map(|b| fulaminas::data::collate::mnist_collate_padded::<NdArray>(b, BATCH_SIZE))
            .enumerate()
        {
            // Execute with data
            // We need to provide data for inputs
            // Tensor::new_input() creates an Input node.

            executor.run(vec![(x.id(), input_val_4d), (target.id(), target_val)]);

            if i % 100 == 0 {
                let loss_val = executor.get_node_data(loss_container.id()).unwrap();
                let loss_scalar = NdArray::to_vec(loss_val)[0];

                println!("Epoch: {}, Iter: {}, Loss: {:.4}", epoch, i, loss_scalar);
            }
        }

        let mut correct_count = 0.0;
        for (input_val_4d, target_val, _mask_val) in mnist_validation_loader
            .iter()
            .map(|b| fulaminas::data::collate::mnist_collate_padded::<NdArray>(b, BATCH_SIZE))
        {
            executor.run(vec![(x.id(), input_val_4d), (target.id(), target_val)]);
            let accu_val = executor.get_node_data(acc_container.id()).unwrap();
            let accu_scalar = NdArray::to_vec(accu_val)[0];
            correct_count += accu_scalar * BATCH_SIZE as f32;
        }
        let accuracy = correct_count / mnist_validation.len() as f32;
        println!("Epoch: {}, Validation Accuracy: {:.4}", epoch, accuracy);
    }
}
