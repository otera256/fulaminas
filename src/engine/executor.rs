use crate::backend::Backend;

use std::collections::HashMap;

use super::node::{Node, NodeId};

#[derive(Debug)]
pub struct Executor<B: Backend> {
    nodes: Vec<Node<B>>,
    training_plan: Vec<NodeId>,
    inference_plan: Vec<NodeId>,
    memory: HashMap<NodeId, B::Tensor>,
}

impl<B: Backend> Executor<B> {
    pub fn new(
        nodes: Vec<Node<B>>,
        initializers: HashMap<NodeId, B::Tensor>,
        training_plan: Vec<NodeId>,
        inference_plan: Vec<NodeId>,
    ) -> Self {
        Self {
            nodes,
            training_plan,
            inference_plan,
            memory: initializers,
        }
    }

    /// 計算グラフを実行します。
    ///
    /// `inputs`には、入力ノード(`Tensor::new_input`で作成)のIDと、実際のデータ(`B::Tensor`)のペアを与えます。
    /// 指定された実行順序(`excusion_order`)に従ってノードを順次処理し、各ノードのデータ(`exec.memory`)を埋めていきます。
    /// Execute the training plan (forward + backward + optimizer updates).
    pub fn step_train(&mut self, inputs: Vec<(NodeId, B::Tensor)>) {
        self.execute_plan(&self.training_plan.clone(), inputs);
    }

    /// Execute the inference plan (forward only).
    pub fn step_inference(&mut self, inputs: Vec<(NodeId, B::Tensor)>) {
        self.execute_plan(&self.inference_plan.clone(), inputs);
    }

    /// Legacy run method for compatibility. Executes training plan.
    /// @deprecated Use step_train or step_inference.
    pub fn run(&mut self, inputs: Vec<(NodeId, B::Tensor)>) {
        // Default to training for now to keep mnist_mlp working until updated
        self.step_train(inputs);
    }

    fn execute_plan(&mut self, plan: &[NodeId], inputs: Vec<(NodeId, B::Tensor)>) {
        // 入力データを各入力ノードにセット
        for (id, data) in inputs {
            self.memory.insert(id, data);
        }

        // トポロジカルソート順に実行
        for &node_id in plan {
            let node = &self.nodes[node_id];

            // 入力データ収集
            let mut input_data = Vec::new();
            for &input_id in &node.inputs {
                if let Some(data) = self.memory.get(&input_id) {
                    input_data.push(data);
                } else {
                    // Control dependency or missing data?
                    // For Assign check later.
                }
            }

            let mut output_data = None;
            let mut assign_target = None;

            let op_name = node.op.name();

            if op_name.starts_with("Assign") {
                // Assign Op: inputs[0] is value, inputs[1] is target tensor.
                // We want to update the memory at target_id with value.
                // Also Assign node acts as Identity for the value, so it returns value.
                let val = input_data[0].clone();
                let target_id = node.inputs[1];
                assign_target = Some((target_id, val.clone()));
                output_data = Some(val); // Assign returns the new value
            } else if op_name == "NoOp" || op_name == "PlaceholderGrad" {
                // Do nothing. Memory might be already populated (Input/Const/Param)
            } else {
                output_data = Some(node.op.forward(&input_data));
            }

            // 計算結果をメモリに保存
            if let Some(data) = output_data {
                self.memory.insert(node_id, data);
            }
            // Assignの処理（ターゲットノードのデータを更新）
            if let Some((target, val)) = assign_target {
                self.memory.insert(target, val);
            }
        }
    }

    /// 指定されたノードの計算結果を取得します。
    pub fn get_node_data(&self, node_id: NodeId) -> Option<&B::Tensor> {
        self.memory.get(&node_id)
    }

    /// 計算グラフをDOT言語形式（Graphviz用）の文字列に変換します。
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph G {\n");
        dot.push_str("    layout=fdp;\n");
        dot.push_str("    node [style=filled];\n");

        for node in &self.nodes {
            let op_name = node.op.name();
            let mut shape_str = "box";
            let mut color = "white";

            if op_name == "Add"
                || op_name == "Sub"
                || op_name == "Mul"
                || op_name == "Div"
                || op_name == "Neg"
                || op_name.starts_with("Powi")
                || op_name == "Sqrt"
            {
                shape_str = "box";
                color = "lightblue";
            } else if op_name == "Matmul"
                || op_name == "Transpose"
                || op_name.starts_with("Reshape")
                || op_name.starts_with("Broadcast")
                || op_name == "Identity"
                || op_name == "OnesLike"
            {
                shape_str = "box";
                color = "lightyellow";
            } else if op_name == "Sigmoid"
                || op_name == "Tanh"
                || op_name == "ReLU"
                || op_name.starts_with("Softmax")
                || op_name == "Exp"
                || op_name == "Log"
            {
                shape_str = "box";
                color = "lightgreen";
            } else if op_name.starts_with("Sum")
                || op_name == "AddN"
                || op_name == "Gt"
                || op_name.starts_with("ArgMax")
                || op_name == "Eq"
            {
                shape_str = "box";
                color = "lightcoral";
            } else if op_name.starts_with("Assign") {
                shape_str = "invhouse";
                color = "orange";
            } else if op_name == "PlaceholderGrad" {
                shape_str = "note";
                color = "plum";
            } else if op_name == "NoOp" {
                match node.role {
                    crate::engine::node::Role::Input => {
                        shape_str = "box";
                        color = "lightgray";
                    }
                    crate::engine::node::Role::LearnableParameter => {
                        shape_str = "octagon";
                        color = "lightyellow";
                    }
                    crate::engine::node::Role::Const => {
                        shape_str = "doublecircle";
                        color = "lightgray";
                    }
                    _ => {
                        shape_str = "box";
                        color = "white";
                    }
                }
            }

            let label_with_shape = if let Some(s) = &node.shape {
                format!("{}\\n{:?}", op_name, s)
            } else {
                op_name.clone()
            };

            dot.push_str(&format!(
                "    {} [label=\"#{}: {}\", shape={}, fillcolor={}];\n",
                node.id, node.id, label_with_shape, shape_str, color
            ));

            for &input in &node.inputs {
                dot.push_str(&format!("    {} -> {};\n", input, node.id));
            }

            // Assignノードの場合はターゲットへ点線のエッジを張る
            if op_name.starts_with("Assign") {
                // target is inputs[1]
                if node.inputs.len() > 1 {
                    let target = node.inputs[1];
                    dot.push_str(&format!(
                        "    {} -> {} [style=dashed, label=\"assign\"];\n",
                        node.id, target
                    ));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }
}
