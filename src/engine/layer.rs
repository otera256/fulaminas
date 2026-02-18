use crate::backend::Backend;

// Tensor<B>に対する演算の集合をまとめておくレイヤーのトレイト
// 例えば、全結合レイヤーや畳み込みレイヤーなどの複雑な演算をまとめて定義するためのトレイト
// モデル全体もこのレイヤーとして定義できるようにする
pub trait Layer<B: Backend> {}