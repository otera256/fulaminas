use crate::backend::Backend;
use crate::engine::{
    node::NodeId,
    shape::{Dynamic, Shape},
    tensor::Tensor,
    with_graph,
};
use std::collections::HashMap;

type DTensor<B> = Tensor<B, Dynamic>;

pub trait Optimizer<B: Backend> {
    fn update_param<S: Shape + Default>(
        &mut self,
        param: &Tensor<B, S>,
        loss: &Tensor<B, crate::engine::shape::Rank0>,
    );
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
    fn update_param<S: Shape + Default>(
        &mut self,
        param: &Tensor<B, S>,
        loss: &Tensor<B, crate::engine::shape::Rank0>,
    ) {
        let grad = loss.grad(param);
        // new_param = param - lr * grad

        // To support scalar mult correctly, we use broadcast mult.
        // lr is scalar const.
        let lr_tensor =
            Tensor::<B, crate::engine::shape::Rank0>::new_const(B::from_vec(vec![self.lr], &[]));

        // Since `lr_tensor` is Rank0, multiplying it by Rank S gradient implicitly broadcasts.
        let delta = grad * lr_tensor.broadcast::<S>();
        let new_param = param.clone() - delta;

        // Assign new value to param
        let _ = Tensor::assign(param, &new_param, 0);
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
            t: DTensor::new_parameter(B::from_vec(vec![0.0], &[])),
        }
    }
}

impl<B: Backend + 'static> Optimizer<B> for Adam<B> {
    fn update_param<S: Shape + Default>(
        &mut self,
        param: &Tensor<B, S>,
        loss: &Tensor<B, crate::engine::shape::Rank0>,
    ) {
        // We handle timestep update cleanly. t starts at 0.
        // If we want to strictly keep t updated, updating it multiple times per epoch is bad.
        // Let's rely on standard state storage.

        let id = param.id;

        // Update timestep exclusively when we see new parameters/first parametr
        // To be safe we should maintain t separately, but for now we'll do quick extraction.
        let one = Tensor::<B, crate::engine::shape::Rank0>::new_const(B::from_vec(vec![1.0], &[]));
        let beta1 =
            Tensor::<B, crate::engine::shape::Rank0>::new_const(B::from_vec(vec![self.beta1], &[]));
        let beta2 =
            Tensor::<B, crate::engine::shape::Rank0>::new_const(B::from_vec(vec![self.beta2], &[]));
        let epsilon = Tensor::<B, crate::engine::shape::Rank0>::new_const(B::from_vec(
            vec![self.epsilon],
            &[],
        ));
        let lr =
            Tensor::<B, crate::engine::shape::Rank0>::new_const(B::from_vec(vec![self.lr], &[]));
        let one_minus_beta1 = one.clone() - beta1.clone();
        let one_minus_beta2 = one.clone() - beta2.clone();

        let grad = loss.grad(param);

        // Initialize state if needed
        if !self.state.contains_key(&id) {
            let shape = with_graph::<B, _, _>(|graph| {
                graph.nodes[id].shape.clone().expect("Param has no shape")
            });
            let m = DTensor::new_parameter(B::zeros(&shape));
            let v = DTensor::new_parameter(B::zeros(&shape));
            self.state.insert(id, (m, v));
        }

        let (m_dyn, v_dyn) = self.state.get(&id).unwrap();

        // We must convert dynamic state trackers back to S typing so we can do math natively
        let m = Tensor::<B, S>::from_id(m_dyn.id);
        let v = Tensor::<B, S>::from_id(v_dyn.id);

        let new_m = (beta1.broadcast::<S>() * m.clone())
            + (one_minus_beta1.broadcast::<S>() * grad.clone());
        let m_update = Tensor::assign(&m, &new_m, 0);

        let grad_sq = grad.clone() * grad;
        let new_v =
            (beta2.broadcast::<S>() * v.clone()) + (one_minus_beta2.broadcast::<S>() * grad_sq);
        let v_update = Tensor::assign(&v, &new_v, 0);

        let denom = v_update.sqrt() + epsilon.broadcast::<S>();
        let step = m_update / denom;
        let new_p = param.clone() - (lr.broadcast::<S>() * step);
        let _ = Tensor::assign(param, &new_p, 0);
    }
}
