use crate::{backend::Backend, engine::{node::{Node, NodeId, NodeType, OpType}, with_graph}};

// Backend::Tensorとは異なり、グラフ構造を普通の演算のように構築できるようにするためのTensor構造体
#[derive(Debug, Clone)]
struct Tensor<B: Backend + 'static> {
    id: NodeId,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend + 'static> Tensor<B> {
    // 入力プレースホルダーを作成するための関数
    fn new_input() -> Tensor<B> {
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Input,
                inputs: Vec::new(),
                data: None, // 入力ノードは構築時には値がない
            });
            new_node_id
        });
        Tensor::<B> { id: node_id, phantom: std::marker::PhantomData }
    }

    // 学習可能なパラメータを作成するための関数
    fn new_parameter(data: B::Tensor) -> Tensor<B> {
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Parameter,
                inputs: Vec::new(),
                data: Some(data), // パラメータノードは構築時に値がある
            });
            new_node_id
        });
        Tensor::<B> { id: node_id, phantom: std::marker::PhantomData }
    }

    // 定数ノードを作成するための関数
    fn new_const(data: B::Tensor) -> Tensor<B> {
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Const,
                inputs: Vec::new(),
                data: Some(data), // 定数ノードは構築時に値がある
            });
            new_node_id
        });
        Tensor::<B> { id: node_id, phantom: std::marker::PhantomData }
    }

    // 新しい演算ノードをグラフに追加するためのヘルパー関数
    fn op(op_type: OpType, inputs: Vec<NodeId>) -> Tensor<B> {
        // グラフビルダーに新しいノードを追加して、そのIDを取得する
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Operation(op_type),
                inputs,
                data: None, // 演算ノードは構築時には値がない
            });
            new_node_id
        });
        Tensor::<B> { id: node_id, phantom: std::marker::PhantomData }
    }

    // 代入ノードをグラフに追加するためのヘルパー関数
    // RNNやオプティマイザーの更新ルールなど、ループ内で変数を更新する必要がある場合に使用する
    fn assign(target: &Tensor<B>, value: &Tensor<B>, depth: usize) -> Tensor<B> {
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Assign { target: target.id, depth },
                inputs: vec![value.id],
                data: None, // 代入ノードは構築時には値がない
            });
            new_node_id
        });
        Tensor::<B> { id: node_id, phantom: std::marker::PhantomData }
    }
}

// 演算のオーバーロード
impl<B: Backend + 'static> std::ops::Add for Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::op(OpType::Add, vec![self.id, rhs.id])
    }
}

impl<B: Backend + 'static> std::ops::Sub for Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::op(OpType::Sub, vec![self.id, rhs.id])
    }
}

impl<B: Backend + 'static> std::ops::Mul for Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::op(OpType::Mul, vec![self.id, rhs.id])
    }
}

impl<B: Backend + 'static> Tensor<B> {
    // メソッドとしても和差積を定義しておくと、演算子オーバーロードと両方使えるので便利
    fn add(self, rhs: Self) -> Self {
        Tensor::op(OpType::Add, vec![self.id, rhs.id])
    }
    fn sub(self, rhs: Self) -> Self {
        Tensor::op(OpType::Sub, vec![self.id, rhs.id])
    }
    fn mul(self, rhs: Self) -> Self {
        Tensor::op(OpType::Mul, vec![self.id, rhs.id])
    }
    // 行列積のための専用関数
    fn matmul(self, rhs: Self) -> Self {
        Tensor::op(OpType::Matmul, vec![self.id, rhs.id])
    }
}