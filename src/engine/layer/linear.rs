use crate::backend::{Backend, Elm};
use crate::engine::layer::Layer;
// use crate::engine::shape::Dynamic; // Not strictly needed if we use fully qualified names
use crate::engine::shape::{Dynamic, Rank1, Rank2};
use crate::engine::tensor::Tensor;

type DTensor<B> = Tensor<B, Dynamic>;

// 重みの初期化戦略
#[derive(Debug, Clone, Copy)]
pub enum InitStrategy {
    XavierNormal,
    XavierUniform,
    HeNormal,
    HeUniform,
}

pub struct Linear<B: Backend + 'static, const I: usize, const O: usize> {
    w: Tensor<B, Rank2<I, O>>,
    b: Tensor<B, Rank1<O>>,
}

impl<B: Backend + 'static, const I: usize, const O: usize> Linear<B, I, O> {
    pub fn new(strategy: InitStrategy) -> Self {
        let (w, b) = match strategy {
            InitStrategy::XavierNormal => {
                let std = (2.0 / (I + O) as Elm).sqrt();
                let w = Tensor::new_parameter(B::random_normal(&[I, O], 0.0, std, None));
                let b = Tensor::new_parameter(B::zeros(&[O]));
                (w, b)
            }
            InitStrategy::XavierUniform => {
                let limit = (6.0 / (I + O) as Elm).sqrt();
                let w = Tensor::new_parameter(B::random_uniform(&[I, O], -limit, limit, None));
                let b = Tensor::new_parameter(B::zeros(&[O]));
                (w, b)
            }
            InitStrategy::HeNormal => {
                let std = (2.0 / I as Elm).sqrt();
                let w = Tensor::new_parameter(B::random_normal(&[I, O], 0.0, std, None));
                let b = Tensor::new_parameter(B::zeros(&[O]));
                (w, b)
            }
            InitStrategy::HeUniform => {
                let limit = (6.0 / I as Elm).sqrt();
                let w = Tensor::new_parameter(B::random_uniform(&[I, O], -limit, limit, None));
                let b = Tensor::new_parameter(B::zeros(&[O]));
                (w, b)
            }
        };

        Self { w, b }
    }

    /// Static forward pass.
    /// x: [Batch, I] -> [Batch, O]
    pub fn forward<const BATCH: usize>(
        &self,
        x: Tensor<B, Rank2<BATCH, I>>,
    ) -> Tensor<B, Rank2<BATCH, O>> {
        // x @ w + b
        // x: (Batch, I), w: (I, O) -> (Batch, O)
        // b: (O) -> Broadcast to (Batch, O) implicitly via Add?
        // We added specialized Add for Rank2 + Rank1.

        let xw = x.matmul_static(self.w.clone());
        xw + self.b.clone()
    }
}

impl<B: Backend + 'static, const I: usize, const O: usize> Layer<B> for Linear<B, I, O> {
    fn forward(&self, x: DTensor<B>) -> DTensor<B> {
        // Dynamic fallback
        // We have to treat self.w and self.b as dynamic for the operation
        // or rely on op_slice to mix them.
        let w_dyn = self.w.to_dynamic();
        let b_dyn = self.b.to_dynamic();

        // x @ w + b
        // x is [Batch, I] (hopefully)
        // w is [I, O]
        // b is [O]

        // Use generic matmul via op which is available on Tensor<B, S> if we use `Tensor::op`?
        // But `Tensor::matmul` is available on any Tensor.
        // Wait, `Tensor::matmul` returns `Self`.
        // So x.matmul(w_dyn) returns DTensor.

        x.matmul(w_dyn) + b_dyn
    }

    fn parameters(&self) -> Vec<DTensor<B>> {
        vec![self.w.to_dynamic(), self.b.to_dynamic()]
    }
}
