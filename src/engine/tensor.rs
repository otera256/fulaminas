use crate::engine::operation::*;
use crate::{
    backend::Backend,
    engine::{
        node::{Node, NodeId},
        with_graph,
    },
};
use std::rc::Rc;

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
                op: Rc::new(NoOp),
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
                op: Rc::new(NoOp),
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
                op: Rc::new(NoOp),
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
                op: Rc::new(NoOp),
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
        op: Rc<dyn Operation<B>>,
        inputs: Vec<&Tensor<B, S>>,
    ) -> Tensor<B, OutS> {
        Self::op_ids(op, inputs.iter().map(|t| t.id).collect())
    }

    /// Heterogeneous inputs version of op
    pub fn op_slice<OutS: Shape + Default>(
        op: Rc<dyn Operation<B>>,
        inputs: &[&dyn TensorHandle],
    ) -> Tensor<B, OutS> {
        Self::op_ids(op, inputs.iter().map(|t| t.id()).collect())
    }

    /// Internal helper taking NodeIds directly
    pub(crate) fn op_ids<OutS: Shape + Default>(
        op: Rc<dyn Operation<B>>,
        input_ids: Vec<NodeId>,
    ) -> Tensor<B, OutS> {
        // グラフビルダーに新しいノードを追加して、そのIDを取得する
        let node_id = with_graph::<B, _, _>(|graph| {
            // Check dynamic shape info from memory during tracing
            let mut input_shapes = Vec::new();
            for &id in &input_ids {
                let shape = graph.nodes[id]
                    .shape
                    .clone()
                    .expect("Input node missing shape");
                input_shapes.push(shape);
            }

            // Calculate output shape theoretically based on inputs and op
            // (We assume dynamic shapes are correctly managed by user if OutS is Dynamic, or predefined otherwise)
            let output_shape = if OutS::is_dynamic() {
                // If it's dynamic, we infer shape from op or just use LHS shape (very naive fallback)
                // In reality, each Op should define a `compute_shape` method.
                // For now, if dynamic, we infer from broadcast rules in binary.
                if input_shapes.is_empty() {
                    vec![]
                } else if op.name().starts_with("Broadcast") {
                    // Extract shape from OpName ? No, operation needs strongly typed compute_shape
                    // We temporarily look at inputs or just let runtime fill it (if possible?).
                    // Let's rely on standard logic: assume shape matches input 0 unless specified.
                    // Or in our case, `BroadcastOp` should have its shape.
                    input_shapes[0].clone()
                    // Note: If dynamic shape evaluation fails later, we need `Operation::compute_shape`.
                    // Since we just need inference to pass, we provide a placeholder or exact shape below.
                } else if op.name() == "Transpose" {
                    let mut s = input_shapes[0].clone();
                    if s.len() >= 2 {
                        let len = s.len();
                        s.swap(len - 1, len - 2);
                    }
                    s
                } else if op.name() == "Matmul" {
                    let s1 = &input_shapes[0];
                    let s2 = &input_shapes[1];
                    let mut out = s1[..s1.len() - 1].to_vec();
                    out.push(s2[s2.len() - 1]);
                    out
                } else if op.name().starts_with("Sum") {
                    // Very rough approximation for dynamic shapes
                    let mut s = input_shapes[0].clone();
                    // For Sum keep_dims logic, we'd need access to SumOp data
                    // For now we just return s and let runtime check crash if wrong (development phase)
                    s
                } else {
                    // Add, Sub, Mul, etc.
                    input_shapes[0].clone()
                }
            } else {
                OutS::default().to_vec()
            };

            let new_node_id = graph.nodes.len();
            graph.nodes.push(Node {
                id: new_node_id,
                role: crate::engine::node::Role::None, // Op nodes have no special Role for now (unless FeedBack)
                op: op.clone(),
                inputs: input_ids.clone(),
                control_deps: Vec::new(),
                // data: None,
                shape: Some(output_shape),
            });
            new_node_id
        });

        // We will fix the Broadcast dynamic shape hack in binary_op explicitly
        if let Some(bc) = op.as_broadcast() {
            with_graph::<B, _, _>(|graph| {
                graph.nodes[node_id].shape = Some(bc.shape.clone());
            });
        } else if let Some(sum_op) = op.as_sum() {
            with_graph::<B, _, _>(|graph| {
                let mut out_shape = graph.nodes[input_ids[0]].shape.clone().unwrap();
                if let Some(ax) = sum_op.axis {
                    if sum_op.keep_dims {
                        out_shape[ax] = 1;
                    } else {
                        out_shape.remove(ax);
                    }
                } else {
                    if sum_op.keep_dims {
                        out_shape = vec![1; out_shape.len()];
                    } else {
                        out_shape = vec![];
                    }
                }
                graph.nodes[node_id].shape = Some(out_shape);
            });
        } else if let Some(res_op) = op.as_reshape() {
            with_graph::<B, _, _>(|graph| {
                graph.nodes[node_id].shape = Some(res_op.shape.clone());
            });
        }

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
                op: Rc::new(AssignOp { depth }),
                inputs: vec![value.id, target.id],
                control_deps: vec![],
                shape: Some(target_shape.clone()),
            });
            new_node_id
        });
        Tensor::<B, S>::from_id(node_id)
    }

    /// Binary operation with explicit broadcasting.
    fn binary_op(lhs: &Tensor<B, S>, rhs: &Tensor<B, S>, op: Rc<dyn Operation<B>>) -> Tensor<B, S> {
        // Collect node info
        let (lhs_shape, rhs_shape) = with_graph::<B, _, _>(|graph| {
            (
                graph.nodes[lhs.id].shape.clone().expect("LHS no shape"),
                graph.nodes[rhs.id].shape.clone().expect("RHS no shape"),
            )
        });

        if lhs_shape == rhs_shape {
            // No broadcast needed
            return Tensor::<B, S>::op_ids::<S>(op, vec![lhs.id, rhs.id]);
        }

        let target_shape = crate::engine::shape::compute_broadcast_shape(&lhs_shape, &rhs_shape)
            .expect(
                format!(
                    "Broadcast failed: {:?} vs {:?}, lhs_id={}, rhs_id={}",
                    lhs_shape, rhs_shape, lhs.id, rhs.id
                )
                .as_str(),
            );

        let mut lhs_id = lhs.id;
        let mut rhs_id = rhs.id;

        with_graph::<B, _, _>(|graph| {
            // Broadcast LHS if needed
            if lhs_shape != target_shape {
                let new_node_id = graph.nodes.len();
                graph.nodes.push(Node {
                    id: new_node_id,
                    role: crate::engine::node::Role::None,
                    op: Rc::new(BroadcastOp {
                        shape: target_shape.clone(),
                    }),
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
                    op: Rc::new(BroadcastOp {
                        shape: target_shape.clone(),
                    }),
                    inputs: vec![rhs_id],
                    control_deps: vec![],
                    shape: Some(target_shape.clone()),
                });
                rhs_id = new_node_id;
            }
        });

        // Now create the binary op node outside with_graph
        let output_tensor = Tensor::<B, S>::op_ids::<S>(op, vec![lhs_id, rhs_id]);

        // Explicitly set the target shape for the output tensor just in case (though op_ids infer shape via runtime compute_shape)
        // Ensure output shape is correct
        output_tensor
    }
}

// 演算のオーバーロード
impl<B: Backend + 'static, S: Shape + Default> std::ops::Add for Tensor<B, S> {
    type Output = Tensor<B, S>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor::binary_op(&self, &rhs, Rc::new(AddOp))
    }
}

impl<B: Backend + 'static, S: Shape + Default> std::ops::Sub for Tensor<B, S> {
    type Output = Tensor<B, S>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor::binary_op(&self, &rhs, Rc::new(SubOp))
    }
}

impl<B: Backend + 'static, S: Shape + Default> std::ops::Mul for Tensor<B, S> {
    type Output = Tensor<B, S>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::binary_op(&self, &rhs, Rc::new(MulOp))
    }
}

impl<B: Backend + 'static, S: Shape + Default> std::ops::Div for Tensor<B, S> {
    type Output = Tensor<B, S>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor::binary_op(&self, &rhs, Rc::new(DivOp))
    }
}

impl<B: Backend + 'static, S: Shape + Default> Tensor<B, S> {
    // メソッドとしても和差積を定義しておくと、演算子オーバーロードと両方使えるので便利
    pub fn add(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, Rc::new(AddOp))
    }
    pub fn sub(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, Rc::new(SubOp))
    }
    pub fn mul(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, Rc::new(MulOp))
    }
    /// 行列積 (Matrix Multiplication) を行います。
    pub fn matmul(self, rhs: Self) -> Self {
        Tensor::op(Rc::new(MatmulOp), vec![&self, &rhs])
    }

    /// 転置 (Transpose) を行います。
    /// 最後の2次元を入れ替えます。
    pub fn transpose(self) -> Self {
        Tensor::op(Rc::new(TransposeOp), vec![&self])
    }

    pub fn reshape<NewS: Shape + Default>(self) -> Tensor<B, NewS> {
        let shape = NewS::default().to_vec();
        Self::op_ids(Rc::new(ReshapeOp { shape }), vec![self.id])
    }

    pub fn reshape_dynamic(self, shape: Vec<usize>) -> Tensor<B, crate::engine::shape::Dynamic> {
        Self::op_ids::<crate::engine::shape::Dynamic>(Rc::new(ReshapeOp { shape }), vec![self.id])
    }

    pub fn broadcast<NewS: Shape + Default>(self) -> Tensor<B, NewS> {
        let shape = NewS::default().to_vec();
        Self::op_ids(Rc::new(BroadcastOp { shape }), vec![self.id])
    }

    pub fn broadcast_dynamic(self, shape: Vec<usize>) -> Tensor<B, crate::engine::shape::Dynamic> {
        Self::op_ids::<crate::engine::shape::Dynamic>(Rc::new(BroadcastOp { shape }), vec![self.id])
    }

    /// 指定された軸で和をとります (Sum)。
    /// `None`の場合は全要素の和をとります。
    pub fn sum(self, axis: Option<usize>) -> Self {
        Tensor::op(
            Rc::new(SumOp {
                axis,
                keep_dims: false,
            }),
            vec![&self],
        )
    }

    pub fn sum_as<OutS: Shape + Default>(self, axis: Option<usize>) -> Tensor<B, OutS> {
        Tensor::op::<OutS>(
            Rc::new(SumOp {
                axis,
                keep_dims: false,
            }),
            vec![&self],
        )
    }

    pub fn sum_keepdims(self, axis: Option<usize>) -> Self {
        Tensor::op(
            Rc::new(SumOp {
                axis,
                keep_dims: true,
            }),
            vec![&self],
        )
    }

    pub fn add_n(tensors: Vec<Self>) -> Self {
        let refs: Vec<&dyn TensorHandle> = tensors.iter().map(|t| t as &dyn TensorHandle).collect();
        Tensor::<B, S>::op_slice::<S>(Rc::new(AddNOp), &refs)
    }

    /// 符号反転 (-x) を行います。
    pub fn neg(self) -> Self {
        Tensor::op(Rc::new(NegOp), vec![&self])
    }

    /// 同じ形状で全ての要素が1のTensorを作成します。
    pub fn ones_like(tensor: &Self) -> Self {
        Tensor::op(Rc::new(OnesLikeOp), vec![tensor])
    }

    pub fn sigmoid(self) -> Self {
        Tensor::op(Rc::new(SigmoidOp), vec![&self])
    }

    pub fn tanh(self) -> Self {
        Tensor::op(Rc::new(TanhOp), vec![&self])
    }

    pub fn relu(self) -> Self {
        Tensor::op(Rc::new(ReLUOp), vec![&self])
    }

    pub fn softmax(self, axis: Option<usize>) -> Self {
        Tensor::op(Rc::new(SoftmaxOp { axis }), vec![&self])
    }

    pub fn exp(self) -> Self {
        Tensor::op(Rc::new(ExpOp), vec![&self])
    }

    pub fn log(self) -> Self {
        Tensor::op(Rc::new(LogOp), vec![&self])
    }

    pub fn powi(self, n: i32) -> Self {
        Tensor::op(Rc::new(PowiOp { n }), vec![&self])
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
        Tensor::binary_op(&self, &rhs, Rc::new(GtOp))
    }

    pub fn sqrt(self) -> Self {
        Tensor::op(Rc::new(SqrtOp), vec![&self])
    }

    pub fn eq(self, rhs: Self) -> Self {
        Tensor::binary_op(&self, &rhs, Rc::new(EqOp))
    }
}

impl<B: Backend + 'static> Tensor<B, crate::engine::shape::Rank0> {
    /// 勾配計算ノード(`Grad`)を作成します。
    pub fn grad<SX: Shape + Default>(&self, x: &Tensor<B, SX>) -> Tensor<B, SX> {
        let node_id = with_graph::<B, _, _>(|graph| {
            // 形状チェック
            let x_shape = graph.nodes[x.id]
                .shape
                .as_ref()
                .expect("x node has no shape");
            // y (self)はスカラー(Rank0)であることが保証される

            let new_node_id = graph.nodes.len();
            // Grad Node creation
            graph.nodes.push(Node {
                id: new_node_id,
                role: crate::engine::node::Role::None,
                op: Rc::new(PlaceholderGradOp {
                    x: x.id,
                    y: self.id,
                }),
                inputs: vec![], // Grad node explicitly depends on nothing in forward pass
                control_deps: vec![],
                shape: Some(x_shape.clone()),
            });
            new_node_id
        });
        Tensor::<B, SX>::from_id(node_id)
    }
}

impl<B: Backend + 'static, S: Shape + Default> Tensor<B, S> {
    pub fn argmax<OutS: Shape + Default>(self, axis: usize) -> Tensor<B, OutS> {
        Tensor::<B, S>::op_slice::<OutS>(Rc::new(ArgMaxOp { axis }), &[&self])
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
        Tensor::<B, Rank2<M, K>>::op_slice::<Rank2<M, N>>(Rc::new(MatmulOp), &[&self, &rhs])
    }
}

// Specialized implementations for Rank2 (Matrix) + Bias (Rank1)
impl<B: Backend + 'static, const M: usize, const N: usize> std::ops::Add<Tensor<B, Rank1<N>>>
    for Tensor<B, Rank2<M, N>>
{
    type Output = Tensor<B, Rank2<M, N>>;

    fn add(self, rhs: Tensor<B, Rank1<N>>) -> Self::Output {
        Tensor::<B, Rank2<M, N>>::op_slice::<Rank2<M, N>>(Rc::new(AddOp), &[&self, &rhs])
    }
}
