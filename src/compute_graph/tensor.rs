use crate::{backend::Backend, compute_graph::{node::{Node, NodeId, NodeType, OpType}, with_graph}};

// Backend::Tensorとは異なり、グラフ構造を普通の演算のように構築できるようにするためのTensor構造体
#[derive(Debug, Clone)]
struct Tensor<B: Backend + 'static> {
    id: NodeId,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend + 'static> Tensor<B> {
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
}