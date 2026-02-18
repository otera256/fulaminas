use crate::{
    backend::Backend,
    engine::{
        node::{Node, NodeId, NodeType, OpType},
        with_graph,
    },
};

// Backend::Tensorとは異なり、グラフ構造を普通の演算のように構築できるようにするためのTensor構造体
/// 計算グラフ上のノードへの参照（ハンドル）を表す構造体。
///
/// `Tensor`はバックエンドの実データ(`B::Tensor`)を直接保持するのではなく、
/// 計算グラフ(`GraphBuilder`)内のノードID(`NodeId`)を保持します。
/// これにより、ユーザーは`Tensor`同士の演算を行うだけで、自動的に計算グラフが構築されます。
#[derive(Debug, Clone, Copy)]
pub struct Tensor<B: Backend + 'static> {
    pub(crate) id: NodeId,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend + 'static> Tensor<B> {
    pub(crate) fn from_id(id: NodeId) -> Self {
        Tensor {
            id,
            phantom: std::marker::PhantomData,
        }
    }

    /// 入力プレースホルダーを作成します。
    ///
    /// 実行時(`Executor::run`)に外部からデータを与えるためのノードです。
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

    /// 学習可能なパラメータを作成します。
    ///
    /// 内部にデータを保持し、学習によって更新される可能性のあるノードです。
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

    /// 定数ノードを作成します。
    ///
    /// パラメータとは異なり、学習によって更新されない固定値を持つノードです。
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

    /// 新しい演算ノードをグラフに追加するための内部ヘルパー関数
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

    /// 代入ノードをグラフに追加します。
    ///
    /// `target`テンソルに`value`テンソルの値を代入する操作を表します。
    /// RNNやオプティマイザーの更新ルールなど、ループ内で変数を更新する必要がある場合に使用します。
    /// `depth`はループのネストレベルなど、実行順序制御に使用される可能性があります。
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
    /// 行列積 (Matrix Multiplication) を行います。
    pub fn matmul(self, rhs: Self) -> Self {
        Tensor::op(OpType::Matmul, vec![&self, &rhs])
    }

    /// 転置 (Transpose) を行います。
    /// 最後の2次元を入れ替えます。
    pub fn transpose(self) -> Self {
        Tensor::op(OpType::Transpose, vec![&self])
    }

    /// 指定された軸で和をとります (Sum)。
    /// `None`の場合は全要素の和をとります。
    pub fn sum(self, axis: Option<usize>) -> Self {
        Tensor::op(OpType::Sum { axis }, vec![&self])
    }

    /// 勾配計算ノード(`Grad`)を作成します。
    ///
    /// `self` (y) を `x` で微分した勾配 (dy/dx) を計算するリクエストをグラフに追加します。
    /// 実際の勾配計算はグラフ構築後(`GraphBuilder::build`)の自動微分フェーズで行われます。
    pub fn grad(&self, x: &Tensor<B>) -> Tensor<B> {
        let node_id = with_graph::<B, _, _>(|graph| {
            // 形状チェック
            let x_shape = graph.nodes[x.id]
                .shape
                .as_ref()
                .expect("x node has no shape");
            // y (self)はスカラーであると仮定する

            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                node_type: NodeType::Grad {
                    x: x.id,
                    y: self.id,
                },
                inputs: vec![], // Grad node explicitly depends on nothing in forward pass, but implicitly on graph
                data: None,
                shape: Some(x_shape.clone()),
            });
            new_node_id
        });
        Tensor::<B> {
            id: node_id,
            phantom: std::marker::PhantomData,
        }
    }

    /// 複数のTensorの和を効率的に計算するノードを作成します。
    pub fn add_n(tensors: Vec<Self>) -> Self {
        let refs: Vec<&Tensor<B>> = tensors.iter().collect();
        Tensor::op(OpType::AddN, refs)
    }

    /// 符号反転 (-x) を行います。
    pub fn neg(self) -> Self {
        Tensor::op(OpType::Neg, vec![&self])
    }

    /// 同じ形状で全ての要素が1のTensorを作成します。
    pub fn ones_like(tensor: &Self) -> Self {
        Tensor::op(OpType::OnesLike, vec![tensor])
    }
}
