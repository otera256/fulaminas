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
        let a = Tensor::<NdArray>::new_input(vec![3]);
        let b = Tensor::<NdArray>::new_input(vec![3]);

        // Output holder
        let output = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[3]));

        let c = a.clone() + b.clone();
        let _assign = Tensor::assign(&output, &c, 0);

        // build takes no arguments now
        let mut executor = build::<NdArray>();
        executor.run(vec![(a, a_data), (b, b_data)]);

        let result = executor.get_node_data(output.id).unwrap();
        assert_eq!(
            NdArray::to_vec(result),
            vec![1.0 + 4.0, 2.0 + 5.0, 3.0 + 6.0]
        );
    }

    #[test]
    fn matmul() {
        let a_data = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b_data = NdArray::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let a = Tensor::<NdArray>::new_input(vec![2, 2]);
        let b = Tensor::<NdArray>::new_input(vec![2, 2]);

        let output = Tensor::<NdArray>::new_parameter(NdArray::zeros(&[2, 2]));

        let c = a.clone().matmul(b.clone());
        let _assign = Tensor::assign(&output, &c, 0);

        let mut executor = build::<NdArray>();
        executor.run(vec![(a, a_data), (b, b_data)]);

        let result = executor.get_node_data(output.id).unwrap();
        assert_eq!(
            NdArray::to_vec(result),
            vec![
                1.0 * 5.0 + 2.0 * 7.0,
                1.0 * 6.0 + 2.0 * 8.0,
                3.0 * 5.0 + 4.0 * 7.0,
                3.0 * 6.0 + 4.0 * 8.0
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_shape_mismatch() {
        let a = Tensor::<NdArray>::new_input(vec![2, 3]);
        let b = Tensor::<NdArray>::new_input(vec![2, 4]);
        // [2, 3] + [2, 4] should panic
        let _c = a + b;
    }

    #[test]
    fn test_assign_loop() {
        // W = W + 1.0 test
        let initial_w = NdArray::from_vec(vec![1.0], &[1]);
        let w = Tensor::<NdArray>::new_parameter(initial_w.clone());
        let c = Tensor::<NdArray>::new_const(NdArray::from_vec(vec![1.0], &[1]));

        let w_new = w.clone() + c.clone();
        let _assign = Tensor::assign(&w, &w_new, 0);

        let mut executor = build::<NdArray>();

        // 1st run
        executor.run(vec![]);
        let res1 = executor.get_node_data(w.id).unwrap();
        assert_eq!(NdArray::to_vec(res1), vec![2.0]);

        // 2nd run
        executor.run(vec![]);
        let res2 = executor.get_node_data(w.id).unwrap();
        assert_eq!(NdArray::to_vec(res2), vec![3.0]);
    }
}
