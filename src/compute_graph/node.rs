use crate::backend::Backend;

pub type NodeId = usize;

#[derive(Clone, Debug)]
pub enum NodeType {
    Input,
    Parameter, // 学習可能なパラメータを表すノード
    Const,
    Operation(OpType),
    Assign {
        target: NodeId,
        depth: usize, // ループの深さを表す。0はループ外、1は最も内側のループ、2はその外側のループ、...となる
    }
}

#[derive(Clone, Debug)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Matmul,
}

pub struct Node<B: Backend> {
    pub id: NodeId,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>,
    // 実行時に値が入る場所
    // 構築時はNoneで、実行時にSome(tensor)になる
    pub data: Option<B::Tensor>,
}