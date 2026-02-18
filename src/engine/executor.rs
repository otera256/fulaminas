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

    /// 計算グラフを実行します。
    ///
    /// `inputs`には、入力ノード(`Tensor::new_input`で作成)に対応するTensorと、実際のデータ(`B::Tensor`)のペアを与えます。
    /// 指定された実行順序(`excusion_order`)に従ってノードを順次処理し、各ノードのデータ(`node.data`)を埋めていきます。
    pub fn run(&mut self, inputs: Vec<(Tensor<B>, B::Tensor)>) {
        // 入力データを各入力ノードにセット
        for (Tensor { id, .. }, data) in inputs {
            self.nodes[id].data = Some(data);
        }

        // トポロジカルソート順に実行
        for &node_id in &self.excusion_order {
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
                        OpType::Sum { axis } => B::sum(input_tensors[0], axis),
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

        for node in &self.nodes {
            let label = match &node.node_type {
                NodeType::Input => format!("Input ({:?})", node.shape),
                NodeType::Parameter => format!("Parameter ({:?})", node.shape),
                NodeType::Const => format!("Const ({:?})", node.shape),
                NodeType::Operation(op) => format!("{:?} ({:?})", op, node.shape),
                NodeType::Assign { target, depth } => {
                    format!("Assign (target={}, depth={})", target, depth)
                }
                NodeType::Grad { x, y } => format!("Grad (x={}, y={})", x, y),
            };

            dot.push_str(&format!(
                "    {} [label=\"{}: {}\"];\n",
                node.id, node.id, label
            ));

            for &input in &node.inputs {
                dot.push_str(&format!("    {} -> {};\n", input, node.id));
            }
        }

        dot.push_str("}\n");
        dot
    }
}
