use super::node::OpType;

pub fn compute_shape(op_type: &OpType, input_shapes: &[&Vec<usize>]) -> Option<Vec<usize>> {
    match op_type {
        OpType::Add | OpType::Sub | OpType::Mul => {
            if input_shapes.len() != 2 {
                return None;
            }
            broadcast_shape(input_shapes[0], input_shapes[1])
        }
        OpType::Matmul => {
            if input_shapes.len() != 2 {
                return None;
            }
            compute_matmul_shape(input_shapes[0], input_shapes[1])
        }
        OpType::Transpose => {
            if input_shapes.len() != 1 {
                return None;
            }
            compute_transpose_shape(input_shapes[0])
        }
        OpType::Sum { axis } => {
            if input_shapes.len() != 1 {
                return None;
            }
            compute_sum_shape(input_shapes[0], *axis)
        }
        OpType::Identity => {
            if input_shapes.len() != 1 {
                return None;
            }
            Some(input_shapes[0].clone())
        }
        OpType::AddN => {
            if input_shapes.is_empty() {
                return None;
            }
            let first_shape = input_shapes[0];
            for shape in input_shapes.iter().skip(1) {
                if *shape != first_shape {
                    return None;
                }
            }
            Some(first_shape.clone())
        }
        OpType::Neg | OpType::OnesLike => {
            if input_shapes.len() != 1 {
                return None;
            }
            Some(input_shapes[0].clone())
        }
        OpType::Sigmoid
        | OpType::Tanh
        | OpType::ReLU
        | OpType::Softmax { .. }
        | OpType::Exp
        | OpType::Log
        | OpType::Powi { .. } => {
            if input_shapes.len() != 1 {
                return None;
            }
            Some(input_shapes[0].clone())
        }
        OpType::Gt => {
            if input_shapes.len() != 2 {
                return None;
            }
            broadcast_shape(input_shapes[0], input_shapes[1])
        }
    }
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
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
            return None;
        }
    }
    Some(result)
}

fn compute_matmul_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let a_ndim = a.len();
    let b_ndim = b.len();

    if a_ndim == 2 && b_ndim == 2 {
        if a[1] != b[0] {
            return None;
        }
        return Some(vec![a[0], b[1]]);
    } else if a_ndim > 2 && b_ndim == 2 {
        // Broadcasting matmul behavior
        // A: [..., M, K], B: [K, N] -> [..., M, N]
        if a[a_ndim - 1] != b[0] {
            return None;
        }
        let mut out_shape = a.to_vec();
        out_shape[a_ndim - 1] = b[1];
        return Some(out_shape);
    }
    // 他のケースは一旦未サポート
    None
}

fn compute_transpose_shape(a: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len();
    if ndim < 2 {
        return Some(a.to_vec());
    }
    let mut shape = a.to_vec();
    shape.swap(ndim - 1, ndim - 2);
    Some(shape)
}

fn compute_sum_shape(a: &[usize], axis: Option<usize>) -> Option<Vec<usize>> {
    match axis {
        Some(ax) => {
            if ax >= a.len() {
                return None;
            }
            let mut shape = a.to_vec();
            shape.remove(ax);
            Some(shape)
        }
        None => Some(vec![]),
    }
}
