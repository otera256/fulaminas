// MNISTとFashionMNISTをMLPで学習させる

use fulaminas::backend::ndarray::NdArray;
use fulaminas::data::loader::DataLoader;
use fulaminas::data::mnist::Mnist;
use fulaminas::engine::build;
use fulaminas::engine::layer::linear::InitStrategy;
use fulaminas::engine::layer::{Layer, linear::Linear};
use fulaminas::engine::loss::CrossEntropyLoss;
use fulaminas::engine::optimizer::{Optimizer, SGD};
use fulaminas::engine::tensor::Tensor;

fn main() {
    let mnist = Mnist::new(
        "./data/mnist",
        true,
        false, // Don't download, use local
        fulaminas::data::mnist::MnistVariant::Mnist,
    )
    .unwrap();
    let mnist_loader = DataLoader::new(&mnist, 256, true);

    // Explicit batch size of 256
    let x = Tensor::<NdArray>::new_input(vec![256, 1, 28, 28]); // 4D Input in graph
    let target = Tensor::<NdArray>::new_input(vec![256, 10]);
    let mask = Tensor::<NdArray>::new_input(vec![256, 1]); // Mask input

    // Flatten: [256, 1, 28, 28] -> [256, 784]
    let x_flat = x.clone().reshape(vec![256, 784]);

    let h1 = Linear::new(784, 256, InitStrategy::XavierNormal).forward(x_flat);
    let h2 = h1.relu();
    let h3 = Linear::new(256, 128, InitStrategy::XavierNormal).forward(h2);
    let h4 = h3.relu();
    let h5 = Linear::new(128, 64, InitStrategy::XavierNormal).forward(h4);
    let h6 = h5.relu();
    let h7 = Linear::new(64, 10, InitStrategy::XavierNormal).forward(h6);
    let y = h7;

    // Use unreduced loss (vector [256])
    let loss_vec =
        CrossEntropyLoss::new_with_reduction(fulaminas::engine::loss::LossReduction::None)
            .forward(y, target.clone());

    // Apply mask: loss_vec * mask -> [256] (zeros for padded samples)
    // Broadcasting: [256] * [256, 1] -> [256, 1]?
    // loss_vec is [256]. mask is [256, 1].
    // We need to reshape loss_vec to [256, 1] or mask to [256].
    // Let's reshape mask to [256].
    let mask_flat = mask.clone().reshape(vec![256]);
    let masked_loss = loss_vec * mask_flat.clone();

    // Mean over valid samples: sum(masked_loss) / sum(mask)
    // sum(mask) gives the count of valid samples.
    // We use a small epsilon to avoid div by zero if batch is empty (unlikely).
    let valid_count = mask_flat.sum(None);
    let total_loss = masked_loss.sum(None);
    let loss = total_loss / valid_count; // Scalar loss

    let mut optimizer = SGD::new(0.1);
    optimizer.step(&loss);

    let mut executor = build::<NdArray>();

    for epoch in 0..10 {
        for (i, (input_val_4d, target_val, mask_val)) in mnist_loader
            .iter()
            .map(|b| fulaminas::data::collate::mnist_collate_padded::<NdArray>(b, 256))
            .enumerate()
        {
            executor.run(vec![
                (x.clone(), input_val_4d),
                (target.clone(), target_val),
                (mask.clone(), mask_val),
            ]);

            if i % 100 == 0 {
                // Loss is a scalar tensor
                let loss_val = executor.get_node_data(loss.id()).unwrap();
                println!(
                    "Epoch: {}, Iter: {}, Loss: {}",
                    epoch,
                    i,
                    loss_val
                        .get(ndarray::IxDyn(&[]))
                        .unwrap_or(&loss_val.first().unwrap())
                );
            }
        }
    }
}
