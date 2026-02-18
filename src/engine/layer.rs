use crate::backend::{Backend, Elm};
use crate::engine::tensor::Tensor;

// 重みの初期化戦略
#[derive(Debug, Clone, Copy)]
pub enum InitStrategy {
    XavierNormal,
    XavierUniform,
    HeNormal,
    HeUniform,
}

// Tensor<B>に対する演算の集合をまとめておくレイヤーのトレイト
// 例えば、全結合レイヤーや畳み込みレイヤーなどの複雑な演算をまとめて定義するためのトレイト
// モデル全体もこのレイヤーとして定義できるようにする
pub trait Layer<B: Backend> {
    fn forward(&self, x: Tensor<B>) -> Tensor<B>;

    // パラメータを返す（学習対象のTensor）
    fn parameters(&self) -> Vec<Tensor<B>>;
}

pub struct Linear<B: Backend + 'static> {
    w: Tensor<B>,
    b: Tensor<B>,
}

impl<B: Backend + 'static> Linear<B> {
    pub fn new(in_features: usize, out_features: usize, strategy: InitStrategy) -> Self {
        let (w, b) = match strategy {
            InitStrategy::XavierNormal => {
                let std = (2.0 / (in_features + out_features) as Elm).sqrt();
                let w = Tensor::new_parameter(B::random_normal(
                    &[in_features, out_features],
                    0.0,
                    std,
                    None,
                ));
                let b = Tensor::new_parameter(B::zeros(&[out_features]));
                (w, b)
            }
            InitStrategy::XavierUniform => {
                let limit = (6.0 / (in_features + out_features) as Elm).sqrt();
                let w = Tensor::new_parameter(B::random_uniform(
                    &[in_features, out_features],
                    -limit,
                    limit,
                    None,
                ));
                let b = Tensor::new_parameter(B::zeros(&[out_features]));
                (w, b)
            }
            InitStrategy::HeNormal => {
                let std = (2.0 / in_features as Elm).sqrt();
                let w = Tensor::new_parameter(B::random_normal(
                    &[in_features, out_features],
                    0.0,
                    std,
                    None,
                ));
                let b = Tensor::new_parameter(B::zeros(&[out_features]));
                (w, b)
            }
            InitStrategy::HeUniform => {
                let limit = (6.0 / in_features as Elm).sqrt();
                let w = Tensor::new_parameter(B::random_uniform(
                    &[in_features, out_features],
                    -limit,
                    limit,
                    None,
                ));
                let b = Tensor::new_parameter(B::zeros(&[out_features]));
                (w, b)
            }
        };

        Self { w, b }
    }
}

impl<B: Backend + 'static> Layer<B> for Linear<B> {
    fn forward(&self, x: Tensor<B>) -> Tensor<B> {
        x.matmul(self.w.clone()) + self.b.clone()
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![self.w.clone(), self.b.clone()]
    }
}
