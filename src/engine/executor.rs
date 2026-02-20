use crate::backend::Backend;

use std::collections::HashMap;

use super::node::{Node, NodeId, OpType};

#[derive(Debug)]
pub struct Executor<B: Backend> {
    nodes: Vec<Node>,
    training_plan: Vec<NodeId>,
    inference_plan: Vec<NodeId>,
    memory: HashMap<NodeId, B::Tensor>,
}

impl<B: Backend> Executor<B> {
    pub fn new(
        nodes: Vec<Node>,
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

        // println!("Execution Order len: {}", self.excusion_order.len());
        // println!("Loss node data before run: {:?}", self.nodes.iter().find(|n| match n.node_type { NodeType::Operation(op) => matches!(op, OpType::Div), _ => false }).and_then(|n| n.data.as_ref()));

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

            // Assign: input[0] is value, input[1] is target (optional/logical).
            // But Assign operation just needs the value to put into memory at TARGET.
            // Wait, Assign { depth } op doesn't carry target info?
            // "Tensor::assign... inputs: vec![value.id, target.id]"
            // So logic needs to be correct.

            let mut output_data = None;
            let mut assign_target = None;

            match &node.op {
                OpType::Add => {
                    output_data = Some(B::add(input_data[0], input_data[1]));
                }
                OpType::Sub => {
                    output_data = Some(B::sub(input_data[0], input_data[1]));
                }
                OpType::Mul => {
                    output_data = Some(B::mul(input_data[0], input_data[1]));
                }
                OpType::Div => {
                    output_data = Some(B::div(input_data[0], input_data[1]));
                }
                OpType::Matmul => {
                    output_data = Some(B::matmul(input_data[0], input_data[1]));
                }
                OpType::Transpose => {
                    output_data = Some(B::transpose(input_data[0]));
                }
                OpType::Reshape { shape } => {
                    output_data = Some(B::reshape(input_data[0], shape));
                }
                OpType::Broadcast { shape } => {
                    output_data = Some(B::broadcast(input_data[0], shape));
                }
                OpType::Sum { axis, keep_dims } => {
                    output_data = Some(B::sum(input_data[0], *axis, *keep_dims));
                }
                OpType::Identity => {
                    output_data = Some(input_data[0].clone());
                }
                OpType::AddN => {
                    let mut sum = input_data[0].clone();
                    for next in &input_data[1..] {
                        sum = B::add(&sum, next);
                    }
                    output_data = Some(sum);
                }
                OpType::Neg => {
                    output_data = Some(B::neg(input_data[0]));
                }
                OpType::OnesLike => {
                    output_data = Some(B::ones_like(input_data[0]));
                }
                OpType::Sigmoid => {
                    output_data = Some(B::sigmoid(input_data[0]));
                }
                OpType::Tanh => {
                    output_data = Some(B::tanh(input_data[0]));
                }
                OpType::ReLU => {
                    output_data = Some(B::relu(input_data[0]));
                }
                OpType::Softmax { axis } => {
                    output_data = Some(B::softmax(input_data[0], *axis));
                }
                OpType::Exp => {
                    output_data = Some(B::exp(input_data[0]));
                }
                OpType::Log => {
                    output_data = Some(B::log(input_data[0]));
                }
                OpType::Powi { n } => {
                    output_data = Some(B::powi(input_data[0], *n));
                }
                OpType::Gt => {
                    output_data = Some(B::gt(input_data[0], input_data[1]));
                }
                OpType::Sqrt => {
                    output_data = Some(B::sqrt(input_data[0]));
                }
                OpType::ArgMax { axis } => {
                    output_data = Some(B::argmax(input_data[0], *axis));
                }
                OpType::Eq => {
                    output_data = Some(B::eq(input_data[0], input_data[1]));
                }
                OpType::Assign { depth: _ } => {
                    // Assign Op: inputs[0] is value, inputs[1] is target tensor.
                    // We want to update the memory at target_id with value.
                    // Also Assign node acts as Identity for the value, so it returns value.
                    let val = input_data[0].clone();

                    // inputs[1] is target tensor. NodeId of target.
                    let target_id = node.inputs[1];

                    assign_target = Some((target_id, val.clone()));
                    output_data = Some(val); // Assign returns the new value
                }
                OpType::NoOp => {
                    // Do nothing. Memory might be already populated (Input/Const/Param)
                }
                OpType::Grad { .. } => {
                    // Do nothing or error? Should not be in execution order usually?
                    // Unless it's a placeholder that wasn't expanded?
                }
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
            let (label, shape, color) = match &node.op {
                OpType::Add
                | OpType::Sub
                | OpType::Mul
                | OpType::Div
                | OpType::Neg
                | OpType::Powi { .. }
                | OpType::Sqrt => (format!("{:?}", node.op), "box", "lightblue"),

                OpType::Matmul
                | OpType::Transpose
                | OpType::Reshape { .. }
                | OpType::Broadcast { .. }
                | OpType::Identity
                | OpType::OnesLike => (format!("{:?}", node.op), "box", "lightyellow"),

                OpType::Sigmoid
                | OpType::Tanh
                | OpType::ReLU
                | OpType::Softmax { .. }
                | OpType::Exp
                | OpType::Log => (format!("{:?}", node.op), "box", "lightgreen"),

                OpType::Sum { .. }
                | OpType::AddN
                | OpType::Gt
                | OpType::ArgMax { .. }
                | OpType::Eq => (format!("{:?}", node.op), "box", "lightcoral"),

                OpType::Assign { depth } => (format!("Assign (d={})", depth), "invhouse", "orange"),
                OpType::Grad { x, y } => (format!("Grad(x={}, y={})", x, y), "note", "plum"),

                OpType::NoOp => match node.role {
                    crate::engine::node::Role::Input => ("Input".to_string(), "box", "lightgray"),
                    crate::engine::node::Role::LearnableParameter => {
                        ("Param".to_string(), "octagon", "lightyellow")
                    }
                    crate::engine::node::Role::Const => {
                        ("Const".to_string(), "doublecircle", "lightgray")
                    }
                    _ => ("NoOp".to_string(), "box", "white"),
                },
            };

            let label_with_shape = if let Some(s) = &node.shape {
                format!("{}\\n{:?}", label, s)
            } else {
                label
            };

            dot.push_str(&format!(
                "    {} [label=\"#{}: {}\", shape={}, fillcolor={}];\n",
                node.id, node.id, label_with_shape, shape, color
            ));

            for &input in &node.inputs {
                dot.push_str(&format!("    {} -> {};\n", input, node.id));
            }

            // Assignノードの場合はターゲットへ点線のエッジを張る
            if let OpType::Assign { .. } = node.op {
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
