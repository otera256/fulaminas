pub mod backend;
pub mod data;
pub mod engine;
pub mod vis;

#[cfg(test)]
mod tests {
    use crate::backend::Backend;
    use crate::backend::ndarray::NdArray;
    use crate::engine::build;
    use crate::engine::shape::Dynamic;
    use crate::engine::tensor::Tensor;

    type DTensor = Tensor<NdArray, Dynamic>;

    #[test]
    fn test_add() {
        let a_data = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b_data = NdArray::from_vec(vec![4.0, 5.0, 6.0], &[3]);
        let a = DTensor::new_input_dynamic(vec![3]);
        let b = DTensor::new_input_dynamic(vec![3]);

        // Output holder
        let output = DTensor::new_parameter(NdArray::zeros(&[3]));

        let c = a.clone() + b.clone();
        let _assign = DTensor::assign(&output, &c, 0);

        // build takes no arguments now
        let mut executor = build::<NdArray>();
        executor.run(vec![(a.id(), a_data), (b.id(), b_data)]);

        let result = executor.get_node_data(output.id()).unwrap();
        assert_eq!(
            NdArray::to_vec(result),
            vec![1.0 + 4.0, 2.0 + 5.0, 3.0 + 6.0]
        );
    }

    #[test]
    fn matmul() {
        let a_data = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b_data = NdArray::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let a = DTensor::new_input_dynamic(vec![2, 2]);
        let b = DTensor::new_input_dynamic(vec![2, 2]);

        let output = DTensor::new_parameter(NdArray::zeros(&[2, 2]));

        // Generic matmul not available for Dynamic, use op
        let c = a.clone().matmul(b.clone());
        let _assign = DTensor::assign(&output, &c, 0);

        let mut executor = build::<NdArray>();
        executor.run(vec![(a.id(), a_data), (b.id(), b_data)]);

        let result = executor.get_node_data(output.id()).unwrap();
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
    #[should_panic(expected = "Broadcast failed")]
    fn test_shape_mismatch() {
        let a = DTensor::new_input_dynamic(vec![2, 3]);
        let b = DTensor::new_input_dynamic(vec![2, 4]);
        // [2, 3] + [2, 4] should panic runtime
        let _c = a + b;
    }

    #[test]
    fn test_assign_loop() {
        // W = W + 1.0 test
        let initial_w = NdArray::from_vec(vec![1.0], &[1]);
        let w = DTensor::new_parameter(initial_w.clone());
        let c = DTensor::new_const(NdArray::from_vec(vec![1.0], &[1]));

        let w_new = w.clone() + c.clone();
        let _assign = DTensor::assign(&w, &w_new, 0);

        let mut executor = build::<NdArray>();

        // 1st run
        executor.run(vec![]);
        let res1 = executor.get_node_data(w.id()).unwrap();
        assert_eq!(NdArray::to_vec(res1), vec![2.0]);

        // 2nd run
        executor.run(vec![]);
        let res2 = executor.get_node_data(w.id()).unwrap();
        assert_eq!(NdArray::to_vec(res2), vec![3.0]);
    }

    #[test]
    fn test_grad_simple() {
        // y = 3 * x^2
        // dy/dx = 6 * x
        // x = 2.0 -> dy/dx = 12.0
        let x_data = NdArray::from_vec(vec![2.0], &[]);
        let x = Tensor::<NdArray, crate::engine::shape::Rank0>::new_input();

        let c3 = Tensor::<NdArray, crate::engine::shape::Rank0>::new_const(NdArray::from_vec(
            vec![3.0],
            &[],
        ));

        let y = c3 * x.clone() * x.clone();
        let grad_x = y.grad(&x);

        // Need to run graph
        let output =
            Tensor::<NdArray, crate::engine::shape::Rank0>::new_parameter(NdArray::zeros(&[]));
        let _assign = Tensor::assign(&output, &grad_x, 0);

        let mut executor = build::<NdArray>();
        executor.run(vec![(x.id(), x_data)]);

        let res = executor.get_node_data(output.id()).unwrap();
        assert_eq!(NdArray::to_vec(res), vec![12.0]);
    }
}
