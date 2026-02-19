use crate::backend::Backend;

/// Collates a batch of MNIST data into backend tensors with padding.
/// Returns (Input Data, Target Data, Mask).
/// Input Data: [Batch, 1, 28, 28] (Padded with 0)
/// Target Data: [Batch, 10] (One-hot encoded, Padded with 0)
/// Mask: [Batch, 1] (1.0 for valid data, 0.0 for padding)
pub fn mnist_collate_padded<B: Backend>(
    batch: Vec<(Vec<f32>, u8)>,
    target_batch_size: usize,
) -> (B::Tensor, B::Tensor, B::Tensor) {
    let current_batch_size = batch.len();
    let (inputs, labels): (Vec<_>, Vec<_>) = batch.into_iter().unzip();

    // Mask setup
    let mut mask_vec = vec![1.0; current_batch_size];
    mask_vec.resize(target_batch_size, 0.0);
    let mask_tensor = B::from_vec(mask_vec, &[target_batch_size, 1]);

    // Stack inputs: [current, 1, 28, 28]
    let mut input_tensors: Vec<B::Tensor> = inputs
        .into_iter()
        .map(|v| B::from_vec(v, &[1, 28, 28]))
        .collect();

    // Pad inputs if needed
    if current_batch_size < target_batch_size {
        let pad_count = target_batch_size - current_batch_size;
        for _ in 0..pad_count {
            input_tensors.push(B::zeros(&[1, 28, 28]));
        }
    }
    let stacked_input = B::stack(&input_tensors, 0);

    // Stack labels: [current, 10] (One-hot)
    let mut label_tensors: Vec<B::Tensor> = labels
        .into_iter()
        .map(|l| {
            let mut one_hot = vec![0.0; 10];
            one_hot[l as usize] = 1.0;
            B::from_vec(one_hot, &[10])
        })
        .collect();

    // Pad labels if needed
    if current_batch_size < target_batch_size {
        let pad_count = target_batch_size - current_batch_size;
        for _ in 0..pad_count {
            label_tensors.push(B::zeros(&[10]));
        }
    }
    let stacked_labels = B::stack(&label_tensors, 0);

    (stacked_input, stacked_labels, mask_tensor)
}
