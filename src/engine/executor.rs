use crate::backend::Backend;

use super::{
    node::{Node, NodeId, NodeType, OpType},
    tensor::Tensor,
};

#[derive(Debug)]
pub struct Executor<B: Backend> {
    nodes: Vec<Node<B>>,
    excusion_order: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl<B: Backend> Executor<B> {
    pub fn new(nodes: Vec<Node<B>>, excusion_order: Vec<NodeId>, outputs: Vec<NodeId>) -> Self {
        Self {
            nodes,
            excusion_order,
            outputs,
        }
    }
    // inputsは、Tensor<B>がどのNodeIdに対応するかを示す
    pub fn run(&mut self, inputs: Vec<(Tensor<B>, B::Tensor)>) -> Vec<B::Tensor> {
        for (Tensor { id, .. }, data) in inputs {
            self.nodes[id].data = Some(data);
        }
        for &node_id in &self.excusion_order {
            let op_info = if let NodeType::Operation(op_type) = &self.nodes[node_id].node_type {
                Some((op_type.clone(), self.nodes[node_id].inputs.clone()))
            } else {
                None
            };

            if let Some((op_type, inputs)) = op_info {
                let input_tensors: Vec<&B::Tensor> = inputs
                    .iter()
                    .map(|&input| self.nodes[input].data.as_ref().unwrap())
                    .collect();

                let output = match op_type {
                    OpType::Add => B::add(input_tensors[0], input_tensors[1]),
                    OpType::Sub => B::sub(input_tensors[0], input_tensors[1]),
                    OpType::Mul => B::mul(input_tensors[0], input_tensors[1]),
                    OpType::Matmul => B::matmul(input_tensors[0], input_tensors[1]),
                };

                self.nodes[node_id].data = Some(output);
            }
        }
        self.outputs
            .iter()
            .map(|&output| self.nodes[output].data.as_ref().unwrap().clone())
            .collect()
    }
}
