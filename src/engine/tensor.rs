use crate::{
    backend::Backend,
    engine::{
        node::{Node, NodeId, NodeType, OpType},
        with_graph,
    },
};

// Backend::Tensorとは異なり、グラフ構造を普通の演算のように構築できるようにするためのTensor構造体
#[derive(Debug, Clone, Copy)]
pub struct Tensor<B: Backend + 'static> {
    pub(crate) id: NodeId,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend + 'static> Tensor<B> {
    // 入力プレースホルダーを作成するための関数
    pub fn new_input(shape: Vec<usize>) -> Tensor<B> {
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Input,
                inputs: Vec::new(),
                data: None, // 入力ノードは構築時には値がない
                shape: Some(shape),
            });
            new_node_id
        });
        Tensor::<B> {
            id: node_id,
            phantom: std::marker::PhantomData,
        }
    }

    // 学習可能なパラメータを作成するための関数
    pub fn new_parameter(data: B::Tensor) -> Tensor<B> {
        let shape = B::shape(&data);
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Parameter,
                inputs: Vec::new(),
                data: Some(data),
                shape: Some(shape),
            });
            new_node_id
        });
        Tensor::<B> {
            id: node_id,
            phantom: std::marker::PhantomData,
        }
    }

    // 定数ノードを作成するための関数
    pub fn new_const(data: B::Tensor) -> Tensor<B> {
        let shape = B::shape(&data);
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Const,
                inputs: Vec::new(),
                data: Some(data),
                shape: Some(shape),
            });
            new_node_id
        });
        Tensor::<B> {
            id: node_id,
            phantom: std::marker::PhantomData,
        }
    }

    // 新しい演算ノードをグラフに追加するためのヘルパー関数
    pub fn op(op_type: OpType, inputs: Vec<&Tensor<B>>) -> Tensor<B> {
        // グラフビルダーに新しいノードを追加して、そのIDを取得する
        let node_id = with_graph::<B, _, _>(|graph| {
            // 入力の形状を取得
            let input_shapes: Vec<Vec<usize>> = inputs
                .iter()
                .map(|t| {
                    graph.nodes[t.id]
                        .shape
                        .clone()
                        .expect("Input node has no shape")
                })
                .collect();

            // 参照のベクタを作成
            let input_shape_refs: Vec<&Vec<usize>> = input_shapes.iter().collect();

            // 形状推論 (engine::shape::compute_shape を使用)
            let output_shape = crate::engine::shape::compute_shape(&op_type, &input_shape_refs)
                .expect("Shape mismatch in operation");

            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Operation(op_type),
                inputs: inputs.iter().map(|&tensor| tensor.id).collect(),
                data: None,
                shape: Some(output_shape),
            });
            new_node_id
        });
        Tensor::<B> {
            id: node_id,
            phantom: std::marker::PhantomData,
        }
    }

    // 代入ノードをグラフに追加するためのヘルパー関数
    // RNNやオプティマイザーの更新ルールなど、ループ内で変数を更新する必要がある場合に使用する
    pub fn assign(target: &Tensor<B>, value: &Tensor<B>, depth: usize) -> Tensor<B> {
        let node_id = with_graph::<B, _, _>(|graph| {
            // 形状チェック
            let target_shape = graph.nodes[target.id]
                .shape
                .as_ref()
                .expect("Target node has no shape");
            let value_shape = graph.nodes[value.id]
                .shape
                .as_ref()
                .expect("Value node has no shape");

            if target_shape != value_shape {
                panic!(
                    "Shape mismatch in assign: target={:?}, value={:?}",
                    target_shape, value_shape
                );
            }

            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Assign {
                    target: target.id,
                    depth,
                },
                inputs: vec![value.id],
                data: None,                        // 代入ノードは構築時には値がない
                shape: Some(target_shape.clone()), // Assignノード自体のShapeはTargetと同じとする
            });
            new_node_id
        });
        Tensor::<B> {
            id: node_id,
            phantom: std::marker::PhantomData,
        }
    }
}

// 演算のオーバーロード
impl<B: Backend + 'static> std::ops::Add for Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::op(OpType::Add, vec![&self, &rhs])
    }
}

impl<B: Backend + 'static> std::ops::Sub for Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::op(OpType::Sub, vec![&self, &rhs])
    }
}

impl<B: Backend + 'static> std::ops::Mul for Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::op(OpType::Mul, vec![&self, &rhs])
    }
}

impl<B: Backend + 'static> Tensor<B> {
    // メソッドとしても和差積を定義しておくと、演算子オーバーロードと両方使えるので便利
    pub fn add(self, rhs: Self) -> Self {
        Tensor::op(OpType::Add, vec![&self, &rhs])
    }
    pub fn sub(self, rhs: Self) -> Self {
        Tensor::op(OpType::Sub, vec![&self, &rhs])
    }
    pub fn mul(self, rhs: Self) -> Self {
        Tensor::op(OpType::Mul, vec![&self, &rhs])
    }
    // 行列積のための専用関数
    pub fn matmul(self, rhs: Self) -> Self {
        Tensor::op(OpType::Matmul, vec![&self, &rhs])
    }
}
