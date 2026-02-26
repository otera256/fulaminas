use std::fmt::Debug;

use crate::backend::Backend;
use crate::engine::shape::Dynamic;
use crate::engine::tensor::Tensor;
use crate::engine::with_graph;

pub type DTensor<B> = Tensor<B, Dynamic>;

/// An atomic operation in the computation graph.
pub trait Operation<B: Backend>: Debug {
    /// Name of the operation, used for debugging and graph visualization.
    fn name(&self) -> String;

    /// Optional downcasting for specific operations
    fn as_broadcast(&self) -> Option<&BroadcastOp> {
        None
    }
    fn as_sum(&self) -> Option<&SumOp> {
        None
    }
    fn as_placeholder_grad(&self) -> Option<&PlaceholderGradOp> {
        None
    }
    fn as_reshape(&self) -> Option<&ReshapeOp> {
        None
    }

    /// Evaluates the operation on the provided backend data.
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor;

    /// Computes the gradients with respect to each input, given the gradient of the output.
    /// `gy`: Gradient of the loss with respect to the output of this operation.
    /// `inputs`: The original Dynamic Tensor inputs to this operation.
    /// Returns a vector of gradients, one for each input.
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>>;
}

#[derive(Debug, Clone)]
pub struct AddOp;
impl<B: Backend + 'static> Operation<B> for AddOp {
    fn name(&self) -> String {
        "Add".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::add(inputs[0], inputs[1])
    }
    fn backward(&self, gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        vec![gy.clone(), gy]
    }
}

#[derive(Debug, Clone)]
pub struct SubOp;
impl<B: Backend + 'static> Operation<B> for SubOp {
    fn name(&self) -> String {
        "Sub".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::sub(inputs[0], inputs[1])
    }
    fn backward(&self, gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        vec![gy.clone(), gy.neg()]
    }
}

#[derive(Debug, Clone)]
pub struct MulOp;
impl<B: Backend + 'static> Operation<B> for MulOp {
    fn name(&self) -> String {
        "Mul".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::mul(inputs[0], inputs[1])
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let a = &inputs[0];
        let b = &inputs[1];
        let ga = gy.clone() * b.clone();
        let gb = gy * a.clone();
        vec![ga, gb]
    }
}

#[derive(Debug, Clone)]
pub struct MatmulOp;
impl<B: Backend + 'static> Operation<B> for MatmulOp {
    fn name(&self) -> String {
        "Matmul".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::matmul(inputs[0], inputs[1])
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let a = &inputs[0];
        let b = &inputs[1];
        let ga = Tensor::op(
            std::rc::Rc::new(MatmulOp),
            vec![&gy, &b.clone().transpose()],
        );
        let gb = Tensor::op(
            std::rc::Rc::new(MatmulOp),
            vec![&a.clone().transpose(), &gy],
        );
        vec![ga, gb]
    }
}

#[derive(Debug, Clone)]
pub struct ReLUOp;
impl<B: Backend + 'static> Operation<B> for ReLUOp {
    fn name(&self) -> String {
        "ReLU".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::relu(inputs[0])
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let x = &inputs[0];
        let zeros = DTensor::new_const(B::zeros(
            with_graph::<B, _, _>(|g| g.nodes[x.id].shape.clone().unwrap()).as_slice(),
        ));
        let mask = x.clone().gt(zeros);
        vec![gy * mask]
    }
}

#[derive(Debug, Clone)]
pub struct DivOp;
impl<B: Backend + 'static> Operation<B> for DivOp {
    fn name(&self) -> String {
        "Div".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::div(inputs[0], inputs[1])
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let a = &inputs[0];
        let b = &inputs[1];
        let inv_b = b.clone().powi(-1);
        let da = gy.clone() * inv_b.clone();

        let b_sq = b.clone().powi(2);
        let neg_a_div_b_sq = a.clone().neg() * b_sq.powi(-1);
        let db = gy.clone() * neg_a_div_b_sq;
        vec![da, db]
    }
}

#[derive(Debug, Clone)]
pub struct NegOp;
impl<B: Backend + 'static> Operation<B> for NegOp {
    fn name(&self) -> String {
        "Neg".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::neg(inputs[0])
    }
    fn backward(&self, gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        vec![gy.neg()]
    }
}

#[derive(Debug, Clone)]
pub struct TransposeOp;
impl<B: Backend + 'static> Operation<B> for TransposeOp {
    fn name(&self) -> String {
        "Transpose".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::transpose(inputs[0])
    }
    fn backward(&self, gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        vec![gy.transpose()]
    }
}

#[derive(Debug, Clone)]
pub struct BroadcastOp {
    pub shape: Vec<usize>,
}
impl<B: Backend + 'static> Operation<B> for BroadcastOp {
    fn name(&self) -> String {
        format!("Broadcast{:?}", self.shape)
    }

    fn as_broadcast(&self) -> Option<&BroadcastOp> {
        Some(self)
    }

    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::broadcast(inputs[0], &self.shape)
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        // Broadcast backwards is sum over broadcasted dims.
        // For now, defer complex logic to handle_broadcast or let Tensor implement it.
        // We replicate handle_broadcast logic here or move it slightly.
        let target = &inputs[0];
        let ga = crate::engine::autodiff::handle_broadcast(&gy, target);
        vec![ga]
    }
}

#[derive(Debug, Clone)]
pub struct SumOp {
    pub axis: Option<usize>,
    pub keep_dims: bool,
}
impl<B: Backend + 'static> Operation<B> for SumOp {
    fn name(&self) -> String {
        format!("Sum(axis={:?}, keep={})", self.axis, self.keep_dims)
    }

    fn as_sum(&self) -> Option<&SumOp> {
        Some(self)
    }

    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::sum(inputs[0], self.axis, self.keep_dims)
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let a = &inputs[0];
        let mut grad = gy.clone();
        if !self.keep_dims {
            if let Some(ax) = self.axis {
                let mut g_shape =
                    with_graph::<B, _, _>(|g| g.nodes[grad.id].shape.clone().unwrap());
                g_shape.insert(ax, 1);
                grad = grad.reshape_dynamic(g_shape);
            }
        }
        let ones = DTensor::<B>::ones_like(a);
        let expanded_grad = ones * grad;
        vec![expanded_grad]
    }
}

#[derive(Debug, Clone)]
pub struct AssignOp {
    pub depth: usize,
}
impl<B: Backend + 'static> Operation<B> for AssignOp {
    fn name(&self) -> String {
        format!("Assign(depth={})", self.depth)
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        // Assign returns the value (inputs[0])
        inputs[0].clone()
    }
    fn backward(&self, _gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        // Assign blocks gradients
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct AddNOp;
impl<B: Backend + 'static> Operation<B> for AddNOp {
    fn name(&self) -> String {
        "AddN".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        let mut sum = inputs[0].clone();
        for next in &inputs[1..] {
            sum = B::add(&sum, next);
        }
        sum
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        inputs.iter().map(|_| gy.clone()).collect()
    }
}

#[derive(Debug, Clone)]
pub struct OnesLikeOp;
impl<B: Backend + 'static> Operation<B> for OnesLikeOp {
    fn name(&self) -> String {
        "OnesLike".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::ones_like(inputs[0])
    }
    // Constants don't usually get gradients, but strictly it's 0. We'll return 0s just in case.
    fn backward(&self, _gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let x = &inputs[0];
        let shape = with_graph::<B, _, _>(|g| g.nodes[x.id].shape.clone().unwrap());
        vec![DTensor::new_const(B::zeros(&shape))]
    }
}

#[derive(Debug, Clone)]
pub struct ReshapeOp {
    pub shape: Vec<usize>,
}
impl<B: Backend + 'static> Operation<B> for ReshapeOp {
    fn name(&self) -> String {
        format!("Reshape{:?}", self.shape)
    }

    fn as_reshape(&self) -> Option<&ReshapeOp> {
        Some(self)
    }

    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::reshape(inputs[0], &self.shape)
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let x = &inputs[0];
        let shape = with_graph::<B, _, _>(|g| g.nodes[x.id].shape.clone().unwrap());
        vec![gy.reshape_dynamic(shape)]
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderGradOp {
    pub x: crate::engine::node::NodeId,
    pub y: crate::engine::node::NodeId,
}
impl<B: Backend + 'static> Operation<B> for PlaceholderGradOp {
    fn name(&self) -> String {
        "PlaceholderGrad".to_string()
    }

    fn as_placeholder_grad(&self) -> Option<&PlaceholderGradOp> {
        Some(self)
    }

    fn forward(&self, _inputs: &[&B::Tensor]) -> B::Tensor {
        panic!("PlaceholderGrad should not be executed")
    }
    fn backward(&self, _gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        panic!("PlaceholderGrad should not be backwarded")
    }
}

#[derive(Debug, Clone)]
pub struct SigmoidOp;
impl<B: Backend + 'static> Operation<B> for SigmoidOp {
    fn name(&self) -> String {
        "Sigmoid".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::sigmoid(inputs[0])
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        // y = sigmoid(x) -> dy/dx = y * (1 - y)
        // Re-calcing y instead of taking it as argument. Ideally we keep forward output...
        // For now, re-evaluate sigmoid(x).
        let x = &inputs[0];
        let y = Tensor::op(std::rc::Rc::new(SigmoidOp), vec![x]);
        let one = DTensor::<B>::ones_like(&y);
        vec![gy * y.clone() * (one - y)]
    }
}

#[derive(Debug, Clone)]
pub struct TanhOp;
impl<B: Backend + 'static> Operation<B> for TanhOp {
    fn name(&self) -> String {
        "Tanh".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::tanh(inputs[0])
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let x = &inputs[0];
        let y = Tensor::op(std::rc::Rc::new(TanhOp), vec![x]);
        let one = DTensor::<B>::ones_like(&y);
        vec![gy * (one - y.powi(2))]
    }
}

#[derive(Debug, Clone)]
pub struct SoftmaxOp {
    pub axis: Option<usize>,
}
impl<B: Backend + 'static> Operation<B> for SoftmaxOp {
    fn name(&self) -> String {
        format!("Softmax(axis={:?})", self.axis)
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::softmax(inputs[0], self.axis)
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let x = &inputs[0];
        let y = Tensor::op(std::rc::Rc::new(SoftmaxOp { axis: self.axis }), vec![x]);
        let y_gy = y.clone() * gy.clone();
        let sum_y_gy = y_gy.sum_keepdims(self.axis);
        vec![y * (gy - sum_y_gy)]
    }
}

#[derive(Debug, Clone)]
pub struct ExpOp;
impl<B: Backend + 'static> Operation<B> for ExpOp {
    fn name(&self) -> String {
        "Exp".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::exp(inputs[0])
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let x = &inputs[0];
        let y = Tensor::op(std::rc::Rc::new(ExpOp), vec![x]);
        vec![gy * y]
    }
}

#[derive(Debug, Clone)]
pub struct LogOp;
impl<B: Backend + 'static> Operation<B> for LogOp {
    fn name(&self) -> String {
        "Log".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::log(inputs[0])
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let x = &inputs[0];
        vec![gy * x.clone().powi(-1)]
    }
}

#[derive(Debug, Clone)]
pub struct PowiOp {
    pub n: i32,
}
impl<B: Backend + 'static> Operation<B> for PowiOp {
    fn name(&self) -> String {
        format!("Powi({})", self.n)
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::powi(inputs[0], self.n)
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        let x = &inputs[0];
        let n_const = DTensor::<B>::new_const(B::from_vec(vec![self.n as f32], &[1]));
        let x_clone = inputs[0].clone();
        let nx_n_minus_1 = x_clone.powi(self.n - 1) * n_const;
        vec![gy * nx_n_minus_1]
    }
}

#[derive(Debug, Clone)]
pub struct GtOp;
impl<B: Backend + 'static> Operation<B> for GtOp {
    fn name(&self) -> String {
        "Gt".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::gt(inputs[0], inputs[1])
    }
    fn backward(&self, _gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        // Non-differentiable
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct EqOp;
impl<B: Backend + 'static> Operation<B> for EqOp {
    fn name(&self) -> String {
        "Eq".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::eq(inputs[0], inputs[1])
    }
    fn backward(&self, _gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        // Non-differentiable
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct SqrtOp;
impl<B: Backend + 'static> Operation<B> for SqrtOp {
    fn name(&self) -> String {
        "Sqrt".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::sqrt(inputs[0])
    }
    fn backward(&self, _gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        // Needed for Adam, we can just return zero or non-diff depending on use case.
        // Adam doesn't backtrack over sqrt anyway usually.
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct ArgMaxOp {
    pub axis: usize,
}
impl<B: Backend + 'static> Operation<B> for ArgMaxOp {
    fn name(&self) -> String {
        format!("ArgMax(axis={})", self.axis)
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        B::argmax(inputs[0], self.axis)
    }
    fn backward(&self, _gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        // Non-differentiable
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct IdentityOp;
impl<B: Backend + 'static> Operation<B> for IdentityOp {
    fn name(&self) -> String {
        "Identity".to_string()
    }
    fn forward(&self, inputs: &[&B::Tensor]) -> B::Tensor {
        inputs[0].clone()
    }
    fn backward(&self, gy: DTensor<B>, inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        inputs.iter().map(|_| gy.clone()).collect()
    }
}

#[derive(Debug, Clone)]
pub struct NoOp;
impl<B: Backend + 'static> Operation<B> for NoOp {
    fn name(&self) -> String {
        "NoOp".to_string()
    }
    fn forward(&self, _inputs: &[&B::Tensor]) -> B::Tensor {
        panic!("NoOp should not be executed directly in forward unless handled exclusively.")
    }
    fn backward(&self, _gy: DTensor<B>, _inputs: &[DTensor<B>]) -> Vec<DTensor<B>> {
        vec![]
    }
}
