use crate::backend::Backend;

use super::{
    node::{Node, NodeId, NodeType, OpType},
    tensor::Tensor,
};

#[derive(Debug)]
pub struct Executor<B: Backend> {
    nodes: Vec<Node<B>>,
    excusion_order: Vec<NodeId>,
}

impl<B: Backend> Executor<B> {
    pub fn new(nodes: Vec<Node<B>>, excusion_order: Vec<NodeId>) -> Self {
        Self {
            nodes,
            excusion_order,
        }
    }
    // inputsは、Tensor<B>がどのNodeIdに対応するかを示す
    pub fn run(&mut self, inputs: Vec<(Tensor<B>, B::Tensor)>) {
        for (Tensor { id, .. }, data) in inputs {
            self.nodes[id].data = Some(data);
        }
        for &node_id in &self.excusion_order {
            let node_type_info = self.nodes[node_id].node_type.clone();
            let mut output_data = None;
            let mut assign_target = None;

            match node_type_info {
                NodeType::Operation(op_type) => {
                    let input_tensors: Vec<&B::Tensor> = self.nodes[node_id]
                        .inputs
                        .iter()
                        .map(|&input| self.nodes[input].data.as_ref().unwrap())
                        .collect();

                    let output = match op_type {
                        OpType::Add => B::add(input_tensors[0], input_tensors[1]),
                        OpType::Sub => B::sub(input_tensors[0], input_tensors[1]),
                        OpType::Mul => B::mul(input_tensors[0], input_tensors[1]),
                        OpType::Matmul => B::matmul(input_tensors[0], input_tensors[1]),
                    };
                    output_data = Some(output);
                }
                NodeType::Assign { target, .. } => {
                    // Assignノードは、input[0]の値をtargetにコピーする
                    let input_id = self.nodes[node_id].inputs[0];
                    let input_val = self.nodes[input_id].data.as_ref().unwrap().clone();
                    assign_target = Some((target, input_val));
                }
                _ => {}
            }

            if let Some(data) = output_data {
                self.nodes[node_id].data = Some(data);
            }
            if let Some((target, val)) = assign_target {
                self.nodes[target].data = Some(val);
            }
        }
    }

    pub fn get_node_data(&self, node_id: NodeId) -> Option<&B::Tensor> {
        self.nodes[node_id].data.as_ref()
    }
}
