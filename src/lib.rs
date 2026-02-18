pub mod backend;
pub mod engine;

#[cfg(test)]
mod tests {
    use crate::backend::ndarray::NdArray;
    use crate::backend::Backend;
    use crate::engine::build;
    use crate::engine::tensor::Tensor;

    #[test]
    fn test_add() {
        let a_data = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b_data = NdArray::from_vec(vec![4.0, 5.0, 6.0], &[3]);
        let a = Tensor::<NdArray>::new_input();
        let b = Tensor::<NdArray>::new_input();
        let c = a.clone() + b.clone();
        let mut executor = build::<NdArray>(vec![c.id]);
        let result = executor.run(vec![(a, a_data), (b, b_data)]);
        assert_eq!(
            NdArray::to_vec(&result[0]),
            vec![1.0 + 4.0, 2.0 + 5.0, 3.0 + 6.0]
        );
    }
}
