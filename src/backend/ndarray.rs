use ndarray::{ArrayD, Dimension, IntoDimension, Ix2};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use rand::{SeedableRng, distributions::Uniform, rngs::StdRng};

use crate::backend::{Backend, Elm};

#[derive(Debug, Clone)]
pub struct NdArray;

impl Backend for NdArray {
    type Tensor = ArrayD<f32>;

    fn zeros(shape: &[usize]) -> Self::Tensor {
        ArrayD::zeros(shape)
    }
    fn ones(shape: &[usize]) -> Self::Tensor {
        ArrayD::ones(shape)
    }
    fn ones_like(tensor: &Self::Tensor) -> Self::Tensor {
        ArrayD::ones(tensor.shape())
    }
    fn random_normal(shape: &[usize], mean: Elm, std: Elm, seed: Option<u64>) -> Self::Tensor {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        ArrayD::random_using(shape, Normal::new(mean, std).unwrap(), &mut rng)
    }
    fn random_uniform(shape: &[usize], low: Elm, high: Elm, seed: Option<u64>) -> Self::Tensor {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        ArrayD::random_using(shape, Uniform::new(low, high), &mut rng)
    }

    fn from_vec(vec: Vec<Elm>, shape: &[usize]) -> Self::Tensor {
        ArrayD::from_shape_vec(shape, vec).unwrap()
    }
    fn to_vec(tensor: &Self::Tensor) -> Vec<Elm> {
        tensor.iter().cloned().collect()
    }

    fn shape(tensor: &Self::Tensor) -> Vec<usize> {
        tensor.shape().to_vec()
    }

    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a + b
    }
    fn sub(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a - b
    }
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a * b
    }
    fn div(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        a / b
    }
    // 実装の簡単のために重みbは2次元行列であると仮定する
    // 例えば, A=[Batch, Time, In], B=[In, Out] -> Output=[Batch, Time, Out]
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        let a_ndim = a.ndim();
        let b_ndim = b.ndim();

        // 両方2次元のとき
        if a_ndim == 2 && b_ndim == 2 {
            let a_view = a.view().into_dimensionality::<Ix2>().unwrap();
            let b_view = b.view().into_dimensionality::<Ix2>().unwrap();
            return a_view.dot(&b_view).into_dyn();
        } else if a_ndim > 2 && b_ndim == 2 {
            let b_view = b.view().into_dimensionality::<Ix2>().unwrap();

            // Aの次元: [..., M, K]
            // Bの次元: [K, N]
            let k_dim = a.shape()[a_ndim - 1];
            assert_eq!(
                k_dim,
                b.shape()[0],
                "Inner dimensions must match for matmul"
            );
            let m_dim = a.shape()[a_ndim - 2];

            // 出力の形状は [..., M, N]
            let mut output_shape = a.shape().to_vec();
            output_shape[a_ndim - 1] = b.shape()[1];

            // 最後の2次元を行列積して、残りの次元はブロードキャストする
            // uninitで高速化できるが、安全性のためにzerosで初期化する
            let mut output = ArrayD::<Elm>::zeros(output_shape);
            let a_chunks = a.exact_chunks((m_dim, k_dim).into_dimension().into_dyn());
            let output_chunks =
                output.exact_chunks_mut((m_dim, b.shape()[1]).into_dimension().into_dyn());
            ndarray::par_azip!((out_sub in output_chunks, a_sub in a_chunks) {
                let a_sub_2d = a_sub.into_dimensionality::<Ix2>().unwrap();

                ndarray::linalg::general_mat_mul(
                    1.0,
                    &a_sub_2d,
                    &b_view,
                    0.0,
                    &mut out_sub.into_dimensionality::<Ix2>().unwrap()
                );
            });

            output
        } else {
            panic!(
                "Unsupported dimensions for matmul: A ndim={}, B ndim={} (shapes: {:?}, {:?})",
                a_ndim,
                b_ndim,
                a.shape(),
                b.shape()
            );
        }
    }

    fn transpose(tensor: &Self::Tensor) -> Self::Tensor {
        tensor.t().to_owned()
    }

    fn stack(tensors: &[Self::Tensor], axis: usize) -> Self::Tensor {
        let views: Vec<_> = tensors.iter().map(|t| t.view()).collect();
        ndarray::stack(ndarray::Axis(axis), &views).unwrap()
    }

    fn reshape(tensor: &Self::Tensor, shape: &[usize]) -> Self::Tensor {
        tensor.clone().into_shape(shape).unwrap()
    }

    fn broadcast(tensor: &Self::Tensor, shape: &[usize]) -> Self::Tensor {
        tensor
            .broadcast(shape)
            .unwrap_or_else(|| {
                panic!(
                    "Broadcast failed: shape={:?}, target={:?}",
                    tensor.shape(),
                    shape
                )
            })
            .to_owned()
    }

    fn sum(a: &Self::Tensor, axis: Option<usize>, keep_dims: bool) -> Self::Tensor {
        match axis {
            Some(ax) => {
                let mut res = a.sum_axis(ndarray::Axis(ax)).into_dyn();
                if keep_dims {
                    // 削除された軸を復活させる
                    // ndarrayのsum_axisは次元を削除する
                    // resのshapeは[A, C] (元が[A, B, C]でaxis=1なら)
                    // ここにaxis=1でサイズ1の次元を挿入したい
                    // insert_axisはviewを返すだけかな？into_shapeを使う。
                    let mut new_shape = res.shape().to_vec();
                    new_shape.insert(ax, 1);
                    res = res.into_shape(new_shape).unwrap();
                }
                res
            }
            None => {
                let val = a.sum();
                if keep_dims {
                    ArrayD::from_elem(vec![1; a.ndim()], val)
                } else {
                    ArrayD::from_elem(vec![], val)
                }
            }
        }
    }

    fn max(a: &Self::Tensor, axis: Option<usize>) -> Self::Tensor {
        match axis {
            Some(ax) => a
                .map_axis(ndarray::Axis(ax), |sub| {
                    sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                })
                .into_dyn(),
            None => ArrayD::from_elem(vec![], a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)),
        }
    }

    fn neg(a: &Self::Tensor) -> Self::Tensor {
        -a
    }

    fn sigmoid(a: &Self::Tensor) -> Self::Tensor {
        a.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn tanh(a: &Self::Tensor) -> Self::Tensor {
        a.mapv(|v| v.tanh())
    }

    fn relu(a: &Self::Tensor) -> Self::Tensor {
        a.mapv(|v| v.max(0.0))
    }

    fn softmax(a: &Self::Tensor, axis: Option<usize>) -> Self::Tensor {
        let axis = axis.unwrap_or_else(|| a.ndim() - 1);
        let mut out = a.clone();
        out.map_axis_mut(ndarray::Axis(axis), |mut view| {
            let max = view.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            view.mapv_inplace(|v| (v - max).exp());
            let sum = view.sum();
            view.mapv_inplace(|v| v / sum);
        });
        out
    }

    fn log(a: &Self::Tensor) -> Self::Tensor {
        a.mapv(|v| v.ln())
    }

    fn exp(a: &Self::Tensor) -> Self::Tensor {
        a.mapv(|v| v.exp())
    }

    fn powi(a: &Self::Tensor, n: i32) -> Self::Tensor {
        a.mapv(|v| v.powi(n))
    }

    fn gt(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        (a - b).mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    fn sqrt(a: &Self::Tensor) -> Self::Tensor {
        a.mapv(|v| v.sqrt())
    }
}
