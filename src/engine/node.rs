use crate::backend::Backend;

/// 計算グラフ内のノードID（インデックス）
pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Input,
    Parameter,
    Const,
    Operation(OpType),
    Assign { target: NodeId, depth: usize },
    Grad { x: NodeId, y: NodeId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    Transpose,
    Sum { axis: Option<usize> },
    Identity,
    AddN,
    Neg,
    OnesLike,
    Sigmoid,
    Tanh,
    ReLU,
    Softmax { axis: Option<usize> },
    Exp,
    Log,
    Powi { n: i32 },
    Gt,
    Sqrt,
}

#[derive(Debug, Clone)]
pub struct Node<B: Backend> {
    pub id: NodeId,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>,
    pub data: Option<B::Tensor>,
    pub shape: Option<Vec<usize>>,
}
