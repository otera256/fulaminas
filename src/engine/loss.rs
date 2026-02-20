use crate::backend::Backend;
use crate::engine::shape::Shape;
use crate::engine::{tensor::Tensor, with_graph};

pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward<B: Backend + 'static, S: Shape + Default>(
        &self,
        pred: Tensor<B, S>,
        target: Tensor<B, S>,
    ) -> Tensor<B, S> {
        let diff = pred - target;
        diff.powi(2)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }

    // logits: [Batch, Classes], target: [Batch, Classes] (one-hot/probs)
    // logits: [Batch, Classes], target: [Batch, Classes] (one-hot/probs)
    // Returns element-wise loss: -target * log(softmax(logits))
    // Shape: [Batch, Classes]
    pub fn forward<B: Backend + 'static, S: Shape + Default>(
        &self,
        logits: Tensor<B, S>,
        target: Tensor<B, S>,
    ) -> Tensor<B, S> {
        // Softmax
        let axis = with_graph::<B, _, _>(|graph| {
            let shape = graph.nodes[logits.id()]
                .shape
                .as_ref()
                .expect("Logits node has no shape");
            shape.len() - 1
        });

        let probs = logits.clone().softmax(Some(axis));
        // Add small epsilon to prevent log(0) explicitly as Dynamic
        let d_probs = probs.to_dynamic();
        let eps =
            Tensor::<B, crate::engine::shape::Dynamic>::new_const(B::from_vec(vec![1e-7], &[]));
        let d_safe_probs = d_probs + eps;
        let d_log_probs = d_safe_probs.log();
        let d_nll = target.to_dynamic() * d_log_probs; // Element-wise: y * log(y_hat).
        let nll = Tensor::<B, S>::from_id(d_nll.id()); // Cast back to S

        nll.neg()
    }
}
