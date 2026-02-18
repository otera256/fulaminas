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

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
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

        let sum_nll = nll.sum(None); // Sum over batch and classes

        // Mean over batch
        let batch_size: f32 = with_graph::<B, _, _>(|graph| {
            let shape = graph.nodes[logits.id()]
                .shape
                .as_ref()
                .expect("Logits node has no shape");
            shape[0] as f32
        });

        let n_inv = Tensor::new_const(B::from_vec(vec![-1.0 / batch_size], &[1])); // Negative sign here

        sum_nll * n_inv
    }
}
