use crate::backend::Backend;
use crate::engine::{
    node::{NodeId, NodeType},
    shape::{Dynamic, Shape},
    tensor::Tensor,
    with_graph,
};
use std::collections::HashMap;

type DTensor<B> = Tensor<B, Dynamic>;

pub trait Optimizer<B: Backend> {
    fn step<S: Shape + Default>(&mut self, loss: &Tensor<B, S>);
}

pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl<B: Backend + 'static> Optimizer<B> for SGD {
    fn step<S: Shape + Default>(&mut self, loss: &Tensor<B, S>) {
        // Collect all parameter nodes as Dynamic Tensors
        let params: Vec<DTensor<B>> = with_graph::<B, _, _>(|graph| {
            graph
                .nodes
                .iter()
                .filter(|n| matches!(n.node_type, NodeType::Parameter))
                .map(|n| Tensor::from_id(n.id))
                .collect()
        });

        for param in params {
            let grad = loss.grad(&param);
            // new_param = param - lr * grad

            // To support scalar mult correctly, we use broadcast mult.
            // lr is scalar const.
            let lr_tensor = DTensor::new_const(B::from_vec(vec![self.lr], &[1]));
            let delta = grad * lr_tensor;
            let new_param = param.clone() - delta;

            // Assign new value to param
            // Assuming depth 0 for now (main training step)
            let _ = DTensor::assign(&param, &new_param, 0);
        }
    }
}

pub struct Adam<B: Backend + 'static> {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    // (m, v) for each parameter
    state: HashMap<NodeId, (DTensor<B>, DTensor<B>)>,
    t: DTensor<B>, // timestep
}

impl<B: Backend + 'static> Adam<B> {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            state: HashMap::new(),
            t: DTensor::new_parameter(B::from_vec(vec![0.0], &[1])),
        }
    }
}

impl<B: Backend + 'static> Optimizer<B> for Adam<B> {
    fn step<S: Shape + Default>(&mut self, loss: &Tensor<B, S>) {
        let params: Vec<DTensor<B>> = with_graph::<B, _, _>(|graph| {
            graph
                .nodes
                .iter()
                .filter(|n| matches!(n.node_type, NodeType::Parameter))
                .map(|n| Tensor::from_id(n.id))
                .collect()
        });

        // Update timestep
        // t = t + 1
        let one = DTensor::new_const(B::from_vec(vec![1.0], &[1]));
        let new_t = self.t.clone() + one.clone();
        let _ = DTensor::assign(&self.t, &new_t, 0);

        // We use the new t for calculation (t=1, 2, ...)

        let beta1 = DTensor::new_const(B::from_vec(vec![self.beta1], &[1]));
        let beta2 = DTensor::new_const(B::from_vec(vec![self.beta2], &[1]));
        let epsilon = DTensor::new_const(B::from_vec(vec![self.epsilon], &[1]));
        let lr = DTensor::new_const(B::from_vec(vec![self.lr], &[1]));
        let one_minus_beta1 = one.clone() - beta1.clone();
        let one_minus_beta2 = one.clone() - beta2.clone();

        for param in params {
            // Check if it's our internal state (t), if so skip
            if param.id == self.t.id {
                continue;
            }

            let id = param.id;
            // Skip m and v themselves if they are Parameters
            if self.state.values().any(|(m, v)| m.id == id || v.id == id) {
                continue;
            }

            let grad = loss.grad(&param);

            // Initialize state if needed
            if !self.state.contains_key(&id) {
                // Get shape
                let shape = with_graph::<B, _, _>(|graph| {
                    graph.nodes[id].shape.clone().expect("Param has no shape")
                });

                // Initialize m = 0, v = 0
                // Use new_parameter so they persist and can be updated
                let m = DTensor::new_parameter(B::zeros(&shape));
                let v = DTensor::new_parameter(B::zeros(&shape));
                self.state.insert(id, (m, v));
            }

            let (m, v) = self.state.get(&id).unwrap();
            let m = m.clone();
            let v = v.clone();

            // Update m
            // m = beta1 * m + (1 - beta1) * grad
            let new_m = beta1.clone() * m.clone() + one_minus_beta1.clone() * grad.clone();
            let _ = DTensor::assign(&m, &new_m, 0);

            // Update v
            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_sq = grad.clone() * grad;
            let new_v = beta2.clone() * v.clone() + one_minus_beta2.clone() * grad_sq;
            let _ = DTensor::assign(&v, &new_v, 0);

            // Update p
            // p -= lr * m / (sqrt(v) + eps)

            let denom = new_v.sqrt() + epsilon.clone();
            let step = new_m / denom;
            let new_p = param.clone() - lr.clone() * step;
            let _ = DTensor::assign(&param, &new_p, 0);
        }
    }
}
