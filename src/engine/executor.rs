use crate::backend::Backend;

use super::node::{Node, NodeId, NodeType, OpType};

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

    /// 計算グラフを実行します。
    ///
    /// `inputs`には、入力ノード(`Tensor::new_input`で作成)のIDと、実際のデータ(`B::Tensor`)のペアを与えます。
    /// 指定された実行順序(`excusion_order`)に従ってノードを順次処理し、各ノードのデータ(`node.data`)を埋めていきます。
    pub fn run(&mut self, inputs: Vec<(NodeId, B::Tensor)>) {
        // 入力データを各入力ノードにセット
        for (id, data) in inputs {
            self.nodes[id].data = Some(data);
        }

        // println!("Execution Order len: {}", self.excusion_order.len());
        // println!("Loss node data before run: {:?}", self.nodes.iter().find(|n| match n.node_type { NodeType::Operation(op) => matches!(op, OpType::Div), _ => false }).and_then(|n| n.data.as_ref()));

        // トポロジカルソート順に実行
        for &node_id in &self.excusion_order {
            // println!(
            //     "Executing node: {} {:?}",
            //     node_id, self.nodes[node_id].node_type
            // );
            let node_type_info = self.nodes[node_id].node_type.clone();
            let mut output_data = None;
            let mut assign_target = None;

            match node_type_info {
                NodeType::Operation(op_type) => {
                    // 入力テンソルのデータを収集
                    let input_tensors: Vec<&B::Tensor> = self.nodes[node_id]
                        .inputs
                        .iter()
                        .map(|&input| self.nodes[input].data.as_ref().unwrap())
                        .collect();

                    // バックエンドを使って演算を実行
                    let output = match op_type {
                        OpType::Add => B::add(input_tensors[0], input_tensors[1]),
                        OpType::Sub => B::sub(input_tensors[0], input_tensors[1]),
                        OpType::Mul => B::mul(input_tensors[0], input_tensors[1]),
                        OpType::Div => B::div(input_tensors[0], input_tensors[1]),
                        OpType::Matmul => B::matmul(input_tensors[0], input_tensors[1]),
                        OpType::Transpose => B::transpose(input_tensors[0]),

                        OpType::Reshape { shape } => B::reshape(input_tensors[0], &shape),
                        OpType::Broadcast { shape } => B::broadcast(input_tensors[0], &shape),
                        OpType::Sum { axis, keep_dims } => {
                            B::sum(input_tensors[0], axis, keep_dims)
                        }
                        OpType::Identity => input_tensors[0].clone(),
                        OpType::AddN => {
                            let mut sum = input_tensors[0].clone();
                            for tensor in input_tensors.iter().skip(1) {
                                sum = B::add(&sum, tensor);
                            }
                            sum
                        }
                        OpType::Neg => B::neg(input_tensors[0]),
                        OpType::OnesLike => B::ones_like(input_tensors[0]),
                        OpType::Sigmoid => B::sigmoid(input_tensors[0]),
                        OpType::Tanh => B::tanh(input_tensors[0]),
                        OpType::ReLU => B::relu(input_tensors[0]),
                        OpType::Softmax { axis } => B::softmax(input_tensors[0], axis),
                        OpType::Exp => B::exp(input_tensors[0]),
                        OpType::Log => B::log(input_tensors[0]),
                        OpType::Powi { n } => B::powi(input_tensors[0], n),
                        OpType::Gt => B::gt(input_tensors[0], input_tensors[1]),
                        OpType::Sqrt => B::sqrt(input_tensors[0]),
                        OpType::ArgMax { axis } => B::argmax(input_tensors[0], axis),
                        OpType::Eq => B::eq(input_tensors[0], input_tensors[1]),
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

            // 計算結果をノードに保存
            if let Some(data) = output_data {
                self.nodes[node_id].data = Some(data);
            }
            // Assignの処理（ターゲットノードのデータを更新）
            if let Some((target, val)) = assign_target {
                self.nodes[target].data = Some(val);
            }
        }
    }

    /// 指定されたノードの計算結果を取得します。
    pub fn get_node_data(&self, node_id: NodeId) -> Option<&B::Tensor> {
        self.nodes[node_id].data.as_ref()
    }

    /// 計算グラフをDOT言語形式（Graphviz用）の文字列に変換します。
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph G {\n");
        dot.push_str("    layout=fdp;\n");
        dot.push_str("    node [style=filled];\n");

        for node in &self.nodes {
            let (label, shape, color) = match &node.node_type {
                NodeType::Input => (format!("Input\\n{:?}", node.shape), "box", "lightgray"),
                NodeType::Parameter => (
                    format!("Parameter\\n{:?}", node.shape),
                    "octagon",
                    "lightyellow",
                ),
                NodeType::Const => (
                    format!("Const\\n{:?}", node.shape),
                    "doublecircle",
                    "lightgray",
                ),
                NodeType::Operation(op) => {
                    let color = match op {
                        OpType::Add
                        | OpType::Sub
                        | OpType::Mul
                        | OpType::Div
                        | OpType::Neg
                        | OpType::Powi { .. }
                        | OpType::Sqrt => "lightblue",
                        OpType::Matmul
                        | OpType::Transpose
                        | OpType::Reshape { .. }
                        | OpType::Broadcast { .. }
                        | OpType::Identity
                        | OpType::OnesLike => "lightyellow",
                        OpType::Sigmoid
                        | OpType::Tanh
                        | OpType::ReLU
                        | OpType::Softmax { .. }
                        | OpType::Exp
                        | OpType::Log => "lightgreen",
                        OpType::Sum { .. }
                        | OpType::AddN
                        | OpType::Gt
                        | OpType::ArgMax { .. }
                        | OpType::Eq => "lightcoral",
                    };
                    (format!("{:?}\\n{:?}", op, node.shape), "ellipse", color)
                }
                NodeType::Assign { target, depth } => (
                    format!("Assign\\n(target={}, depth={})", target, depth),
                    "invhouse",
                    "orange",
                ),
                NodeType::Grad { x, y } => (format!("Grad\\n(x={}, y={})", x, y), "note", "plum"),
            };

            dot.push_str(&format!(
                "    {} [label=\"#{}: {}\", shape={}, fillcolor={}];\n",
                node.id, node.id, label, shape, color
            ));

            for &input in &node.inputs {
                dot.push_str(&format!("    {} -> {};\n", input, node.id));
            }

            // Assignノードの場合はターゲットへ点線のエッジを張る
            if let NodeType::Assign { target, .. } = node.node_type {
                dot.push_str(&format!(
                    "    {} -> {} [style=dashed, label=\"assign\"];\n",
                    node.id, target
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }
}
