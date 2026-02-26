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

pub fn compute_broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
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
