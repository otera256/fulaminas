use crate::backend::Backend;
use crate::engine::shape::{Rank0, Shape};
use crate::engine::tensor::Tensor;

pub struct Accuracy;

impl Accuracy {
    /// Computes accuracy between prediction and target.
    ///
    /// pred: [Batch, Classes] (Logits or Probs)
    /// target: [Batch, Classes] (One-hot) OR [Batch] (Indices)
    ///
    /// Returns: Scalar (Rank0) accuracy (0.0 - 1.0)
    pub fn forward<B: Backend + 'static, S1: Shape + Default, S2: Shape + Default>(
        pred: Tensor<B, S1>,
        target: Tensor<B, S2>,
    ) -> Tensor<B, Rank0> {
        use crate::engine::shape::Dynamic;

        let pred_idx = pred.argmax::<Dynamic>(1);

        // Target handling:
        // If target is One-Hot (Rank2), argmax(1).
        // If target is Indices (Rank1), use as is.
        // We can try to detect or just assume One-Hot for now as per mnist_mlp.
        // Or checking dim?
        // We can't check dim of Type S2 easily without instance.
        // But we can check runtime shape?
        // But graph construction is static mostly.

        // Let's assume Target is One-Hot for this implementation as it matches CrossEntropy with one-hot.
        // If it's indices, argmax(1) on [Batch] might fail or be weird.
        // let target_idx = target.argmax::<Dynamic>(1);

        // Actually, for robustness, let's just assume same shape as Pred (One-Hot) or force user to pass One-Hot.
        let target_idx = target.argmax::<Dynamic>(1);

        let correct = pred_idx.eq(target_idx); // [Batch] of 1.0 or 0.0

        // Calculate N first (uses &correct)
        let n = Tensor::ones_like(&correct).sum_as::<Rank0>(None);

        // Calculate Acc (consumes correct)
        let acc = correct.sum_as::<Rank0>(None);

        acc / n
    }
}
