use crate::backend::Backend;
use crate::engine::{tensor::Tensor, with_graph};

pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward<B: Backend + 'static>(&self, pred: Tensor<B>, target: Tensor<B>) -> Tensor<B> {
        let diff = pred.clone() - target;
        let sq_diff = diff.powi(2);
        let sum_sq = sq_diff.sum(None);

        // Calculate N from shape
        let n: f32 = with_graph::<B, _, _>(|graph| {
            let shape = graph.nodes[pred.id()]
                .shape
                .as_ref()
                .expect("Pred node has no shape");
            shape.iter().product::<usize>() as f32
        });

        // Create 1/N constant
        let n_inv = Tensor::new_const(B::from_vec(vec![1.0 / n], &[1]));

        // Mean = Sum * (1/N)
        sum_sq * n_inv
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LossReduction {
    Mean,
    Sum,
    None,
}

pub struct CrossEntropyLoss {
    reduction: LossReduction,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            reduction: LossReduction::Mean,
        }
    }

    pub fn new_with_reduction(reduction: LossReduction) -> Self {
        Self { reduction }
    }

    // logits: [Batch, Classes], target: [Batch, Classes] (one-hot/probs)
    pub fn forward<B: Backend + 'static>(&self, logits: Tensor<B>, target: Tensor<B>) -> Tensor<B> {
        // Softmax
        let axis = with_graph::<B, _, _>(|graph| {
            let shape = graph.nodes[logits.id()]
                .shape
                .as_ref()
                .expect("Logits node has no shape");
            shape.len() - 1
        });

        let probs = logits.clone().softmax(Some(axis));
        // Add small epsilon to prevent log(0) if needed, but for now assuming stable or distinct logits
        // For stability in production, LogSoftmax op is better.
        let log_probs = probs.log();
        let nll = target * log_probs; // Element-wise: y * log(y_hat)

        // Sum over classes (axis=1) to get loss per sample
        // nll is [Batch, Classes]. Sum(axis=1) -> [Batch]
        // But our Sum reduces dimension by default.
        // We want [Batch] (vector).
        // If we use sum(axis=1, keep_dims=false), we get [Batch].
        // But nll is negative log likelihood * target.
        // Target is one-hot -> only one element is non-zero.
        // So sum over classes picks the correct class log-prob.
        // Result is NEGATIVE.
        let sum_nll = nll.sum(Some(axis));

        // sum_nll is [Batch]. Values are negative (log(p)).
        // We want positive loss. -sum_nll.
        let loss_per_sample = sum_nll.neg();

        match self.reduction {
            LossReduction::None => loss_per_sample,
            LossReduction::Sum => loss_per_sample.sum(None),
            LossReduction::Mean => {
                // Mean over batch
                let batch_size: f32 = with_graph::<B, _, _>(|graph| {
                    let shape = graph.nodes[logits.id()]
                        .shape
                        .as_ref()
                        .expect("Logits node has no shape");
                    shape[0] as f32
                });
                let n_inv = Tensor::new_const(B::from_vec(vec![1.0 / batch_size], &[1]));
                loss_per_sample.sum(None) * n_inv
            }
        }
    }
}
