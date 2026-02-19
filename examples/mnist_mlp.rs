// MNIST and FashionMNIST MLP with Static Shapes

use fulaminas::backend::Backend;
use fulaminas::backend::ndarray::NdArray;
use fulaminas::data::loader::DataLoader;
use fulaminas::data::mnist::Mnist;
use fulaminas::engine::build;
use fulaminas::engine::layer::linear::{InitStrategy, Linear};
use fulaminas::engine::loss::CrossEntropyLoss;
use fulaminas::engine::optimizer::{Optimizer, SGD};
use fulaminas::engine::shape::{Rank0, Rank1, Rank2, Rank4};
use fulaminas::engine::tensor::Tensor;

// Define constraints for network
const BATCH_SIZE: usize = 256;
const IN_FEATURES: usize = 784;
const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 128;
const HIDDEN_3: usize = 64;
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
    let layer1 = Linear::<NdArray, IN_FEATURES, HIDDEN_1>::new(InitStrategy::XavierNormal);
    let layer2 = Linear::<NdArray, HIDDEN_1, HIDDEN_2>::new(InitStrategy::XavierNormal);
    let layer3 = Linear::<NdArray, HIDDEN_2, HIDDEN_3>::new(InitStrategy::XavierNormal);
    let layer4 = Linear::<NdArray, HIDDEN_3, CLASSES>::new(InitStrategy::XavierNormal);

    // Forward pass
    // Note: linear.forward(x) expects Tensor<B, Rank2<Batch, In>>
    let h1 = layer1.forward(x_flat).relu();
    let h2 = layer2.forward(h1).relu();
    let h3 = layer3.forward(h2).relu();
    let y = layer4.forward(h3);

    println!("Graph built with Static Shapes.");

    // Loss
    // Returns [Batch, Classes] (element-wise nll)
    let elem_loss = CrossEntropyLoss::new().forward(y, target.clone());

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

    let mut optimizer = SGD::new(0.1);
    // optimizer.step requires Tensor<B, S>
    optimizer.step(&loss);

    let mut executor = build::<NdArray>();

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
                // Loss is a scalar tensor
                let loss_val = executor.get_node_data(loss.id()).unwrap();
                println!(
                    "Epoch: {}, Iter: {}, Loss: {}",
                    epoch,
                    i,
                    loss_val.to_string()
                );
            }
        }
    }
}
