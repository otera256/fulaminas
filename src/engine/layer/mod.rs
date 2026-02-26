use crate::backend::Backend;
use crate::engine::tensor::Tensor;

pub mod activations;
pub mod linear;
pub mod mlp;

use crate::engine::shape::Dynamic;

type DTensor<B> = Tensor<B, Dynamic>;

// Tensor<B>に対する演算の集合をまとめておくレイヤーのトレイト
// 例えば、全結合レイヤーや畳み込みレイヤーなどの複雑な演算をまとめて定義するためのトレイト
// モデル全体もこのレイヤーとして定義できるようにする
pub trait Layer<B: Backend> {
    fn forward(&self, x: DTensor<B>) -> DTensor<B>;
}
