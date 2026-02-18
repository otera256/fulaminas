use crate::backend::Backend;

pub type NodeId = usize;

#[derive(Clone, Debug)]
pub enum NodeType {
    /// 外部からの入力データを受け取るノード
    Input,
    /// 学習可能なパラメータを表すノード
    Parameter,
    /// 定数値を表すノード
    Const,
    /// 演算を表すノード
    Operation(OpType),
    /// 変数への代入を表すノード
    Assign {
        target: NodeId,
        // 外部への出力または、モデル内部のループの深さを表す。
        // depthが小さいほど内側のループ
        depth: usize,
    },
    /// 自動微分を行うためのノード
    /// yをxで微分した結果 (dy/dx) を表す
    Grad { x: NodeId, y: NodeId },
}

#[derive(Clone, Debug)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    Transpose,
    Sum {
        axis: Option<usize>,
    },
    AddN,
    Neg,
    OnesLike,
    Sigmoid,
    Tanh,
    ReLU,
    Softmax {
        axis: Option<usize>,
    },
    Exp,
    Log,
    Powi {
        n: i32,
    },
    Gt,
    Sqrt,
    /// 逆伝播時の勾配置換用（入力をそのまま出力する）
    Identity,
}

#[derive(Clone, Debug)]
pub struct Node<B: Backend> {
    pub id: NodeId,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>,
    /// 実行時に計算結果の値が格納される場所
    /// グラフ構築時はNoneで、Executorによる実行時にSome(tensor)になる
    pub data: Option<B::Tensor>,
    pub shape: Option<Vec<usize>>,
}
