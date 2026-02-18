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
        // 外部への出力または、モデル内部のループの深さを表す。
        // depthが小さいほど内側のループ
        // どこかしらで出力の深さを指定する
        // モデルの学習自体もdepthが最大となるときの出力としてみなせる（する必要があるかは要検討）
        depth: usize,
    },
    // 自動微分を行うためのノード
    // yをxで微分した結果を表す
    Grad {
        x: NodeId,
        y: NodeId,
    },
}

#[derive(Clone, Debug)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Matmul,
    Transpose,
    Sum { axis: Option<usize> },
    // 逆伝播時の勾配置換用
    Identity,
}

#[derive(Clone, Debug)]
pub struct Node<B: Backend> {
    pub id: NodeId,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>,
    // 実行時に値が入る場所
    // 構築時はNoneで、実行時にSome(tensor)になる
    pub data: Option<B::Tensor>,
    pub shape: Option<Vec<usize>>,
}
