use super::node::OpType;

pub trait Shape: Clone + std::fmt::Debug + Send + Sync + 'static {
    type Array;
    fn as_array(&self) -> Self::Array;
    fn to_vec(&self) -> Vec<usize>;
    fn is_dynamic() -> bool {
        false
    }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Dynamic;
impl Shape for Dynamic {
    type Array = Vec<usize>;
    fn as_array(&self) -> Self::Array {
        vec![]
    }
    fn to_vec(&self) -> Vec<usize> {
        vec![]
    }
    fn is_dynamic() -> bool {
        true
    }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Rank0;
impl Shape for Rank0 {
    type Array = [usize; 0];
    fn as_array(&self) -> Self::Array {
        []
    }
    fn to_vec(&self) -> Vec<usize> {
        vec![]
    }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Rank1<const M: usize>;
impl<const M: usize> Shape for Rank1<M> {
    type Array = [usize; 1];
    fn as_array(&self) -> Self::Array {
        [M]
    }
    fn to_vec(&self) -> Vec<usize> {
        vec![M]
    }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Rank2<const M: usize, const N: usize>;
impl<const M: usize, const N: usize> Shape for Rank2<M, N> {
    type Array = [usize; 2];
    fn as_array(&self) -> Self::Array {
        [M, N]
    }
    fn to_vec(&self) -> Vec<usize> {
        vec![M, N]
    }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Rank3<const B: usize, const M: usize, const N: usize>;
impl<const B: usize, const M: usize, const N: usize> Shape for Rank3<B, M, N> {
    type Array = [usize; 3];
    fn as_array(&self) -> Self::Array {
        [B, M, N]
    }
    fn to_vec(&self) -> Vec<usize> {
        vec![B, M, N]
    }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Rank4<const B: usize, const C: usize, const H: usize, const W: usize>;
impl<const B: usize, const C: usize, const H: usize, const W: usize> Shape for Rank4<B, C, H, W> {
    type Array = [usize; 4];
    fn as_array(&self) -> Self::Array {
        [B, C, H, W]
    }
    fn to_vec(&self) -> Vec<usize> {
        vec![B, C, H, W]
    }
}

pub fn compute_shape(op_type: &OpType, input_shapes: &[&Vec<usize>]) -> Result<Vec<usize>, String> {
    match op_type {
        OpType::Add | OpType::Sub => {
            if input_shapes.len() != 2 {
                return Err(format!(
                    "Add/Sub requires 2 inputs, got {}",
                    input_shapes.len()
                ));
            }
            broadcast_shape(input_shapes[0], input_shapes[1])
        }
        OpType::Mul => {
            if input_shapes.len() != 2 {
                return Err(format!("Mul requires 2 inputs, got {}", input_shapes.len()));
            }
            broadcast_shape(input_shapes[0], input_shapes[1])
        }
        OpType::Div => {
            if input_shapes.len() != 2 {
                return Err(format!("Div requires 2 inputs, got {}", input_shapes.len()));
            }
            broadcast_shape(input_shapes[0], input_shapes[1])
        }
        OpType::Matmul => {
            if input_shapes.len() != 2 {
                return Err(format!(
                    "Matmul requires 2 inputs, got {}",
                    input_shapes.len()
                ));
            }
            compute_matmul_shape(input_shapes[0], input_shapes[1])
        }
        OpType::Transpose => {
            if input_shapes.len() != 1 {
                return Err(format!(
                    "Transpose requires 1 input, got {}",
                    input_shapes.len()
                ));
            }
            compute_transpose_shape(input_shapes[0])
        }
        OpType::Reshape { shape } => {
            if input_shapes.len() != 1 {
                return Err(format!(
                    "Reshape requires 1 input, got {}",
                    input_shapes.len()
                ));
            }
            // Validate total elements
            let input_total: usize = input_shapes[0].iter().product();
            let output_total: usize = shape.iter().product();
            if input_total != output_total {
                return Err(format!(
                    "Reshape mismatch: input total {} != output total {}",
                    input_total, output_total
                ));
            }
            Ok(shape.clone())
        }
        OpType::Broadcast { shape } => {
            if input_shapes.len() != 1 {
                return Err(format!(
                    "Broadcast requires 1 input, got {}",
                    input_shapes.len()
                ));
            }
            let input_shape = input_shapes[0];
            let common = broadcast_shape(input_shape, shape)?;
            if &common != shape {
                return Err(format!(
                    "Cannot broadcast {:?} to target {:?}. Result would be {:?}",
                    input_shape, shape, common
                ));
            }
            Ok(shape.clone())
        }
        OpType::Sum { axis, keep_dims } => {
            if input_shapes.len() != 1 {
                return Err(format!("Sum requires 1 input, got {}", input_shapes.len()));
            }
            compute_sum_shape(input_shapes[0], *axis, *keep_dims)
        }
        OpType::Identity => {
            if input_shapes.len() != 1 {
                return Err(format!(
                    "Identity requires 1 input, got {}",
                    input_shapes.len()
                ));
            }
            Ok(input_shapes[0].clone())
        }
        OpType::AddN => {
            if input_shapes.is_empty() {
                return Err("AddN requires at least 1 input".to_string());
            }
            let first_shape = input_shapes[0];
            for (i, shape) in input_shapes.iter().enumerate().skip(1) {
                if *shape != first_shape {
                    return Err(format!(
                        "AddN shape mismatch at index {}: expected {:?}, got {:?}",
                        i, first_shape, shape
                    ));
                }
            }
            Ok(first_shape.clone())
        }
        OpType::Neg | OpType::OnesLike => {
            if input_shapes.len() != 1 {
                return Err(format!(
                    "Neg/OnesLike requires 1 input, got {}",
                    input_shapes.len()
                ));
            }
            Ok(input_shapes[0].clone())
        }
        OpType::Sigmoid
        | OpType::Tanh
        | OpType::ReLU
        | OpType::Softmax { .. }
        | OpType::Exp
        | OpType::Log
        | OpType::Powi { .. } => {
            if input_shapes.len() != 1 {
                return Err(format!(
                    "Activation/Math op requires 1 input, got {}",
                    input_shapes.len()
                ));
            }
            Ok(input_shapes[0].clone())
        }
        OpType::Gt => {
            if input_shapes.len() != 2 {
                return Err(format!("Gt requires 2 inputs, got {}", input_shapes.len()));
            }
            broadcast_shape(input_shapes[0], input_shapes[1])
        }
        OpType::Sqrt => {
            if input_shapes.len() != 1 {
                return Err(format!("Sqrt requires 1 input, got {}", input_shapes.len()));
            }
            Ok(input_shapes[0].clone())
        }
        OpType::ArgMax { axis } => {
            if input_shapes.len() != 1 {
                return Err(format!(
                    "ArgMax requires 1 input, got {}",
                    input_shapes.len()
                ));
            }
            // Argmax reduces the dimension at axis
            compute_sum_shape(input_shapes[0], Some(*axis), false)
        }
        OpType::Eq => {
            if input_shapes.len() != 2 {
                return Err(format!("Eq requires 2 inputs, got {}", input_shapes.len()));
            }
            broadcast_shape(input_shapes[0], input_shapes[1])
        }
    }
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let a_len = a.len();
    let b_len = b.len();
    let max_len = a_len.max(b_len);
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_dim = if i < max_len - a_len {
            1
        } else {
            a[i - (max_len - a_len)]
        };
        let b_dim = if i < max_len - b_len {
            1
        } else {
            b[i - (max_len - b_len)]
        };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return Err(format!(
                "Broadcast error: dimension mismatch at index {} (from right): {} vs {} (shapes: {:?}, {:?})",
                max_len - i - 1,
                a_dim,
                b_dim,
                a,
                b
            ));
        }
    }
    Ok(result)
}

fn compute_matmul_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let a_ndim = a.len();
    let b_ndim = b.len();

    if a_ndim == 2 && b_ndim == 2 {
        if a[1] != b[0] {
            return Err(format!(
                "Matmul shape mismatch: inner dimensions MUST match. {:?} x {:?} ({} != {})",
                a, b, a[1], b[0]
            ));
        }
        return Ok(vec![a[0], b[1]]);
    } else if a_ndim > 2 && b_ndim == 2 {
        // Broadcasting matmul behavior
        // A: [..., M, K], B: [K, N] -> [..., M, N]
        if a[a_ndim - 1] != b[0] {
            return Err(format!(
                "Matmul shape mismatch with broadcasting: inner dimensions MUST match. {:?} x {:?} ({} != {})",
                a,
                b,
                a[a_ndim - 1],
                b[0]
            ));
        }
        let mut out_shape = a.to_vec();
        out_shape[a_ndim - 1] = b[1];
        return Ok(out_shape);
    }

    if a_ndim == 1 {
        return Err(format!(
            "Matmul requires at least 2 dimensions (e.g. [Batch, In]). Input A has 1 dimension {:?}. Consider using shape [1, {}] or similar.",
            a, a[0]
        ));
    }

    Err(format!(
        "Unsupported dimensions for matmul: A ndim={}, B ndim={} (shapes: {:?}, {:?})",
        a_ndim, b_ndim, a, b
    ))
}

fn compute_transpose_shape(a: &[usize]) -> Result<Vec<usize>, String> {
    let ndim = a.len();
    if ndim < 2 {
        // Scalar or 1D vector transpose is no-op or identity
        return Ok(a.to_vec());
    }
    let mut shape = a.to_vec();
    shape.swap(ndim - 1, ndim - 2);
    Ok(shape)
}

fn compute_sum_shape(
    a: &[usize],
    axis: Option<usize>,
    keep_dims: bool,
) -> Result<Vec<usize>, String> {
    match axis {
        Some(ax) => {
            if ax >= a.len() {
                return Err(format!("Sum axis {} out of bounds for shape {:?}", ax, a));
            }
            let mut shape = a.to_vec();
            if keep_dims {
                shape[ax] = 1;
            } else {
                shape.remove(ax);
            }
            Ok(shape)
        }
        None => {
            if keep_dims {
                Ok(vec![1; a.len()])
            } else {
                Ok(vec![])
            }
        }
    }
}
