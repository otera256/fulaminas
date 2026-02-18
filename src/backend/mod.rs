use std::fmt::Debug;

pub mod ndarray;

type Elm = f32;

pub trait Backend: Clone + Debug {
    type Tensor: Clone + Debug;

    fn zeros(shape: &[usize]) -> Self::Tensor;
    fn ones(shape: &[usize]) -> Self::Tensor;
    fn ones_like(tensor: &Self::Tensor) -> Self::Tensor;
    fn random_normal(shape: &[usize], mean: Elm, std: Elm, seed: Option<u64>) -> Self::Tensor;
    fn random_uniform(shape: &[usize], low: Elm, high: Elm, seed: Option<u64>) -> Self::Tensor;

    // CPU配列からの作成
    fn from_vec(vec: Vec<Elm>, shape: &[usize]) -> Self::Tensor;
    // CPU配列への変換(基本的に重い処理となる)
    fn to_vec(tensor: &Self::Tensor) -> Vec<Elm>;

    fn shape(tensor: &Self::Tensor) -> Vec<usize>;

    // 基本的な演算(全て新しいTensorを返す)
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn sub(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor; // 要素ごとの積
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor; // 行列積

    fn transpose(tensor: &Self::Tensor) -> Self::Tensor;

    fn relu(tensor: &Self::Tensor) -> Self::Tensor;

    fn sum(a: &Self::Tensor, axis: Option<usize>) -> Self::Tensor;
    fn max(a: &Self::Tensor, axis: Option<usize>) -> Self::Tensor;

    fn neg(a: &Self::Tensor) -> Self::Tensor;
}
