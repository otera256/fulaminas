use crate::{
    backend::Backend,
    engine::{
        node::{Node, NodeId, OpType},
        with_graph,
    },
};

use crate::engine::shape::{Rank1, Rank2, Shape};

// Backend::Tensorとは異なり、グラフ構造を普通の演算のように構築できるようにするためのTensor構造体
/// 計算グラフ上のノードへの参照（ハンドル）を表す構造体。
///
/// `Tensor`はバックエンドの実データ(`B::Tensor`)を直接保持するのではなく、
/// 計算グラフ(`GraphBuilder`)内のノードID(`NodeId`)を保持します。
/// これにより、ユーザーは`Tensor`同士の演算を行うだけで、自動的に計算グラフが構築されます。
#[derive(Debug, Clone, Copy)]
pub struct Tensor<B: Backend + 'static, S: Shape> {
    pub(crate) id: NodeId,
    phantom: std::marker::PhantomData<(B, S)>,
}

pub trait TensorHandle {
    fn id(&self) -> NodeId;
}

impl<B: Backend + 'static, S: Shape> TensorHandle for Tensor<B, S> {
    fn id(&self) -> NodeId {
        self.id
    }
}

impl<B: Backend + 'static, S: Shape + Default> Tensor<B, S> {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub(crate) fn from_id(id: NodeId) -> Self {
        Tensor {
            id,
            phantom: std::marker::PhantomData,
        }
    }

    /// Converts this tensor to a Dynamic tensor.
    pub fn to_dynamic(&self) -> Tensor<B, crate::engine::shape::Dynamic> {
        Tensor::from_id(self.id)
    }

    /// 入力プレースホルダーを作成します。
    ///
    /// 実行時(`Executor::run`)に外部からデータを与えるためのノードです。
    pub fn new_input() -> Self {
        let shape = Self::default_shape();
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                role: crate::engine::node::Role::Input,
                op: OpType::NoOp,
                inputs: Vec::new(),
                control_deps: Vec::new(),
                shape: Some(shape),
            });
            new_node_id
        });
        Self::from_id(node_id)
    }

    pub fn new_input_dynamic(shape: Vec<usize>) -> Tensor<B, crate::engine::shape::Dynamic> {
        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                role: crate::engine::node::Role::Input,
                op: OpType::NoOp,
                inputs: Vec::new(),
                control_deps: Vec::new(),
                shape: Some(shape),
            });
            new_node_id
        });
        Tensor::from_id(node_id)
    }

    fn default_shape() -> Vec<usize> {
        // dummy instance to call trait method? No, trait methods are on instances.
        // We need S to be default constructible or just use types.
        // Shape trait has no static method for shape.
        // But we added to_vec(&self).
        // Since S: Default (Rank structs derive Default), we can do:
        S::default().to_vec()
    }

    /// 学習可能なパラメータを作成します。
    ///
    /// 内部にデータを保持し、学習によって更新される可能性のあるノードです。
    pub fn new_parameter(data: B::Tensor) -> Self {
        let shape = B::shape(&data);
        if !S::is_dynamic() {
            let s_shape = Self::default_shape();
            if shape != s_shape {
                panic!(
                    "Parameter data shape {:?} does not match Generic Shape {:?}",
                    shape, s_shape
                );
            }
        }

        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                role: crate::engine::node::Role::LearnableParameter,
                op: OpType::NoOp,
                inputs: Vec::new(),
                control_deps: Vec::new(),
                shape: Some(shape),
            });
            // Store initial/current value in initializers
            graph.initializers.insert(new_node_id, data);
            new_node_id
        });
        Self::from_id(node_id)
    }

    /// 定数ノードを作成します。
    ///
    /// パラメータとは異なり、学習によって更新されない固定値を持つノードです。
    pub fn new_const(data: B::Tensor) -> Self {
        let shape = B::shape(&data);
        if !S::is_dynamic() {
            let s_shape = Self::default_shape();
            if shape != s_shape {
                panic!(
                    "Const data shape {:?} does not match Generic Shape {:?}",
                    shape, s_shape
                );
            }
        }

        let node_id = with_graph::<B, _, _>(|graph| {
            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                role: crate::engine::node::Role::Const,
                op: OpType::NoOp,
                inputs: Vec::new(),
                control_deps: Vec::new(),
                shape: Some(shape),
            });
            graph.initializers.insert(new_node_id, data);
            new_node_id
        });
        Self::from_id(node_id)
    }

    /// 新しい演算ノードをグラフに追加するための内部ヘルパー関数
    ///
    /// OutputShapeを指定してノードを作成します。
    pub fn op<OutS: Shape + Default>(
        op_type: OpType,
        inputs: Vec<&Tensor<B, S>>,
    ) -> Tensor<B, OutS> {
        Self::op_ids(op_type, inputs.iter().map(|t| t.id).collect())
    }

    /// Heterogeneous inputs version of op
    pub fn op_slice<OutS: Shape + Default>(
        op_type: OpType,
        inputs: &[&dyn TensorHandle],
    ) -> Tensor<B, OutS> {
        Self::op_ids(op_type, inputs.iter().map(|t| t.id()).collect())
    }

    /// Internal helper taking NodeIds directly
    pub(crate) fn op_ids<OutS: Shape + Default>(
        op_type: OpType,
        input_ids: Vec<NodeId>,
    ) -> Tensor<B, OutS> {
        // グラフビルダーに新しいノードを追加して、そのIDを取得する
        let node_id = with_graph::<B, _, _>(|graph| {
            // 形状推論 (runtime validation)
            let input_shapes: Vec<Vec<usize>> = input_ids
                .iter()
                .map(|&id| {
                    graph.nodes[id]
                        .shape
                        .clone()
                        .expect("Input node has no shape")
                })
                .collect();
            let input_shape_refs: Vec<&Vec<usize>> = input_shapes.iter().collect();

            let output_shape = crate::engine::shape::compute_shape(&op_type, &input_shape_refs)
                .unwrap_or_else(|e| panic!("Shape mismatch in operation: {}", e));

            // Verify that validated runtime shape matches OutS
            if !OutS::is_dynamic() {
                let expected_shape = OutS::default().to_vec();
                if output_shape != expected_shape {
                    panic!(
                        "Runtime shape {:?} != Type shape {:?}",
                        output_shape, expected_shape
                    );
                }
            }

            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                role: crate::engine::node::Role::None, // Op nodes have no special Role for now (unless FeedBack)
                op: op_type,
                inputs: input_ids,
                control_deps: Vec::new(),
                // data: None,
                shape: Some(output_shape),
            });
            new_node_id
        });
        Tensor::<B, OutS>::from_id(node_id)
    }

    /// 代入ノードをグラフに追加します。
    pub fn assign(target: &Tensor<B, S>, value: &Tensor<B, S>, depth: usize) -> Tensor<B, S> {
        let node_id = with_graph::<B, _, _>(|graph| {
            // 形状チェック (Runtime)
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
                role: crate::engine::node::Role::None,
                op: OpType::Assign { depth },
                inputs: vec![value.id, target.id],
                control_deps: vec![],
                shape: Some(target_shape.clone()),
            });
            new_node_id
        });
        Tensor::<B, S>::from_id(node_id)
    }

    /// Binary operation with explicit broadcasting.
    fn binary_op(lhs: &Tensor<B, S>, rhs: &Tensor<B, S>, op_type: OpType) -> Tensor<B, S> {
        // Collect node info
        let (lhs_shape, rhs_shape) = with_graph::<B, _, _>(|graph| {
            (
                graph.nodes[lhs.id].shape.clone().expect("LHS no shape"),
                graph.nodes[rhs.id].shape.clone().expect("RHS no shape"),
            )
        });

        if lhs_shape == rhs_shape {
            // No broadcast needed
            return Tensor::<B, S>::op_ids::<S>(op_type, vec![lhs.id, rhs.id]);
        }

        let target_shape = crate::engine::shape::compute_broadcast_shape(&lhs_shape, &rhs_shape)
            .expect("Broadcast failed");

        let mut lhs_id = lhs.id;
        let mut rhs_id = rhs.id;

        with_graph::<B, _, _>(|graph| {
            // Broadcast LHS if needed
            if lhs_shape != target_shape {
                let new_node_id = graph.nodes.len();
                graph.nodes.push(Node {
                    id: new_node_id,
                    role: crate::engine::node::Role::None,
                    op: OpType::Broadcast {
                        shape: target_shape.clone(),
                    },
                    inputs: vec![lhs_id],
                    control_deps: vec![],
                    shape: Some(target_shape.clone()),
                });
                lhs_id = new_node_id;
            }

            // Broadcast RHS if needed
            if rhs_shape != target_shape {
                let new_node_id = graph.nodes.len();
                graph.nodes.push(Node {
                    id: new_node_id,
                    role: crate::engine::node::Role::None,
                    op: OpType::Broadcast {
                        shape: target_shape.clone(),
                    },
                    inputs: vec![rhs_id],
                    control_deps: vec![],
                    shape: Some(target_shape.clone()),
                });
                rhs_id = new_node_id;
            }
        });

        // Now create the binary op node outside with_graph
        let output_tensor = Tensor::<B, S>::op_ids::<S>(op_type, vec![lhs_id, rhs_id]);

        // Explicitly set the target shape for the output tensor just in case (though op_ids infer shape via runtime compute_shape)
        // Ensure output shape is correct
        output_tensor
    }
}

// 演算のオーバーロード
impl<B: Backend + 'static, S: Shape + Default> std::ops::Add for Tensor<B, S> {
    type Output = Tensor<B, S>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::binary_op(&self, &rhs, OpType::Add)
    }
}

impl<B: Backend + 'static, S: Shape + Default> std::ops::Sub for Tensor<B, S> {
    type Output = Tensor<B, S>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::binary_op(&self, &rhs, OpType::Sub)
    }
}

impl<B: Backend + 'static, S: Shape + Default> std::ops::Mul for Tensor<B, S> {
    type Output = Tensor<B, S>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::binary_op(&self, &rhs, OpType::Mul)
    }
}

impl<B: Backend + 'static, S: Shape + Default> std::ops::Div for Tensor<B, S> {
    type Output = Tensor<B, S>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::binary_op(&self, &rhs, OpType::Div)
    }
}

impl<B: Backend + 'static, S: Shape + Default> Tensor<B, S> {
    // メソッドとしても和差積を定義しておくと、演算子オーバーロードと両方使えるので便利
    pub fn add(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, OpType::Add)
    }
    pub fn sub(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, OpType::Sub)
    }
    pub fn mul(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, OpType::Mul)
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

    pub fn reshape<NewS: Shape + Default>(self) -> Tensor<B, NewS> {
        let shape = NewS::default().to_vec();
        Self::op_ids(OpType::Reshape { shape }, vec![self.id])
    }

    pub fn reshape_dynamic(self, shape: Vec<usize>) -> Tensor<B, crate::engine::shape::Dynamic> {
        Self::op_ids::<crate::engine::shape::Dynamic>(OpType::Reshape { shape }, vec![self.id])
    }

    pub fn broadcast<NewS: Shape + Default>(self) -> Tensor<B, NewS> {
        let shape = NewS::default().to_vec();
        Self::op_ids(OpType::Broadcast { shape }, vec![self.id])
    }

    pub fn broadcast_dynamic(self, shape: Vec<usize>) -> Tensor<B, crate::engine::shape::Dynamic> {
        Self::op_ids::<crate::engine::shape::Dynamic>(OpType::Broadcast { shape }, vec![self.id])
    }

    /// 指定された軸で和をとります (Sum)。
    /// `None`の場合は全要素の和をとります。
    pub fn sum(self, axis: Option<usize>) -> Self {
        Tensor::op(
            OpType::Sum {
                axis,
                keep_dims: false,
            },
            vec![&self],
        )
    }

    pub fn sum_as<OutS: Shape + Default>(self, axis: Option<usize>) -> Tensor<B, OutS> {
        Tensor::op::<OutS>(
            OpType::Sum {
                axis,
                keep_dims: false,
            },
            vec![&self],
        )
    }

    pub fn sum_keepdims(self, axis: Option<usize>) -> Self {
        Tensor::op(
            OpType::Sum {
                axis,
                keep_dims: true,
            },
            vec![&self],
        )
    }

    /// 勾配計算ノード(`Grad`)を作成します。
    pub fn grad<SX: Shape + Default>(&self, x: &Tensor<B, SX>) -> Tensor<B, SX> {
        let node_id = with_graph::<B, _, _>(|graph| {
            // 形状チェック
            let x_shape = graph.nodes[x.id]
                .shape
                .as_ref()
                .expect("x node has no shape");
            // y (self)はスカラーであると仮定する

            let new_node_id = graph.nodes.len();
            // Grad Node creation
            graph.nodes.push(Node {
                id: new_node_id,
                role: crate::engine::node::Role::None,
                op: OpType::Grad {
                    x: x.id,
                    y: self.id,
                },
                inputs: vec![], // Grad node explicitly depends on nothing in forward pass
                control_deps: vec![],
                shape: Some(x_shape.clone()),
            });
            new_node_id
        });
        Tensor::<B, SX>::from_id(node_id)
    }

    /// 複数のTensorの和を効率的に計算するノードを作成します。
    pub fn add_n(tensors: Vec<Self>) -> Self {
        let refs: Vec<&Tensor<B, S>> = tensors.iter().collect();
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

    pub fn sigmoid(self) -> Self {
        Tensor::op(OpType::Sigmoid, vec![&self])
    }

    pub fn tanh(self) -> Self {
        Tensor::op(OpType::Tanh, vec![&self])
    }

    pub fn relu(self) -> Self {
        Tensor::op(OpType::ReLU, vec![&self])
    }

    pub fn softmax(self, axis: Option<usize>) -> Self {
        Tensor::op(OpType::Softmax { axis }, vec![&self])
    }

    pub fn exp(self) -> Self {
        Tensor::op(OpType::Exp, vec![&self])
    }

    pub fn log(self) -> Self {
        Tensor::op(OpType::Log, vec![&self])
    }

    pub fn powi(self, n: i32) -> Self {
        Tensor::op(OpType::Powi { n }, vec![&self])
    }

    /// Swish activation: x * sigmoid(x)
    pub fn swish(self) -> Self {
        self.clone() * self.sigmoid()
    }

    /// Softplus activation: log(1 + exp(x))
    pub fn softplus(self) -> Self {
        let one = Tensor::ones_like(&self);
        (one + self.exp()).log()
    }

    pub fn gt(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, OpType::Gt)
    }

    pub fn sqrt(self) -> Self {
        Tensor::op(OpType::Sqrt, vec![&self])
    }

    pub fn eq(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, OpType::Eq)
    }
}

impl<B: Backend + 'static, S: Shape + Default> Tensor<B, S> {
    pub fn argmax<OutS: Shape + Default>(self, axis: usize) -> Tensor<B, OutS> {
        Tensor::<B, S>::op_slice::<OutS>(OpType::ArgMax { axis }, &[&self])
    }
}

// Specialized implementations for Rank2 (Matrix)
impl<B: Backend + 'static, const M: usize, const K: usize> Tensor<B, Rank2<M, K>> {
    /// Statically checked matrix multiplication.
    /// (M, K) x (K, N) -> (M, N)
    pub fn matmul_static<const N: usize>(
        self,
        rhs: Tensor<B, Rank2<K, N>>,
    ) -> Tensor<B, Rank2<M, N>> {
        Tensor::<B, Rank2<M, K>>::op_slice::<Rank2<M, N>>(OpType::Matmul, &[&self, &rhs])
    }
}

// Specialized implementations for Rank2 (Matrix) + Bias (Rank1)
impl<B: Backend + 'static, const M: usize, const N: usize> std::ops::Add<Tensor<B, Rank1<N>>>
    for Tensor<B, Rank2<M, N>>
{
    type Output = Tensor<B, Rank2<M, N>>;

    fn add(self, rhs: Tensor<B, Rank1<N>>) -> Self::Output {
        Tensor::<B, Rank2<M, N>>::op_slice::<Rank2<M, N>>(OpType::Add, &[&self, &rhs])
    }
}
