// 自動微分(Automatic Differentiation)を行うためのモジュール

use std::collections::HashMap;

use crate::backend::Backend;
use crate::engine::{
    node::{Node, NodeId},
    shape::Dynamic,
    tensor::Tensor,
    with_graph,
};

// AutoDiff内部では形状が動的に決定されるため、Dynamic Shapeを使用する
type DTensor<B> = Tensor<B, Dynamic>;

/// 計算グラフを展開し、逆伝播（バックプロパゲーション）のためのノードを追加します。
pub fn expand_graph<B: Backend + 'static>() {
    // 1. 計算すべき勾配の特定
    // Instead of OpType::Grad, let's look for nodes that have Role::Feedback?
    // Or we will update tensor.rs `grad` to use a special Operation for this temporarily.
    // Let's assume there is a `crate::engine::operation::GradOp` or we look at `Role::Feedback`.
    // We will check that when we fix tensor.rs. Let's assume a GradOp exists for a second,
    // or better, we check if the operation's name is "PlaceholderGrad".

    let (match_grads, global_roots) = with_graph::<B, _, _>(|graph| {
        let mut grads_to_process: HashMap<NodeId, Vec<(NodeId, NodeId)>> = HashMap::new();
        let mut global_roots = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            if let Some(grad_op) = node.op.as_placeholder_grad() {
                let x = grad_op.x;
                let y = grad_op.y;
                grads_to_process.entry(y).or_default().push((x, i));
                if !global_roots.contains(&y) {
                    global_roots.push(y);
                }
            }
        }
        (grads_to_process, global_roots)
    });

    if match_grads.is_empty() {
        return;
    }

    // 2. 関連するグラフ全体のトポロジカルソートを取得
    let forward_order = with_graph::<B, _, _>(|graph| graph.topological_sort(&global_roots));

    // 3. 各yのグループごとにバックワードパスを実行
    for (y_root, x_targets) in match_grads {
        let mut node_grads: HashMap<NodeId, Vec<DTensor<B>>> = HashMap::new();

        // 初期勾配 dy/dy = 1 を設定
        let y_tensor = DTensor::<B>::from_id(y_root);
        let ones_tensor = DTensor::<B>::ones_like(&y_tensor);
        node_grads.entry(y_root).or_default().push(ones_tensor);

        // トポロジカルソートの逆順（出力から入力へ）でイテレーション
        for &node_id in forward_order.iter().rev() {
            // 現在のノードに対する勾配を収集
            let final_grad = if let Some(grads) = node_grads.remove(&node_id) {
                if grads.len() == 1 {
                    grads[0].clone()
                } else {
                    DTensor::<B>::add_n(grads)
                }
            } else {
                continue;
            };

            // Gradノードへの接続
            with_graph::<B, _, _>(|graph| {
                for &(x, grad_node_id) in &x_targets {
                    if x == node_id {
                        // Replace Grad placeholder with Identity connected to final_grad
                        graph.nodes[grad_node_id].op =
                            std::rc::Rc::new(crate::engine::operation::IdentityOp);
                        graph.nodes[grad_node_id].inputs = vec![final_grad.id];
                    }
                }
            });

            // バックプロパゲーション（入力への勾配伝播）
            let (op, inputs) = with_graph::<B, _, _>(|graph| {
                let node = &graph.nodes[node_id];
                (node.op.clone(), node.inputs.clone())
            });

            let input_tensors: Vec<DTensor<B>> =
                inputs.iter().map(|&id| DTensor::<B>::from_id(id)).collect();

            // Perform dynamic dispatch backward
            let input_grads = op.backward(final_grad, &input_tensors);

            // Accumulate returned gradients into the map
            // Note: input_grads should match the order and length of input_tensors.
            // Some operations might not return gradients for all inputs (e.g. Assign).
            for (i, grad) in input_grads.into_iter().enumerate() {
                if i < input_tensors.len() {
                    node_grads
                        .entry(input_tensors[i].id)
                        .or_default()
                        .push(grad);
                }
            }
        }

        // Deal with unconnected gradients (parameters that didn't receive gradients)
        with_graph::<B, _, _>(|graph| {
            for &(x, grad_node_id) in &x_targets {
                if graph.nodes[grad_node_id].op.name() == "PlaceholderGrad" {
                    let shape = graph.nodes[x].shape.clone().expect("Shape missing");
                    let zeros = B::zeros(&shape);

                    let zeros_id = graph.nodes.len();
                    graph.nodes.push(Node {
                        id: zeros_id,
                        role: crate::engine::node::Role::Const, // Zeros is Const
                        op: std::rc::Rc::new(crate::engine::operation::NoOp),
                        inputs: vec![],
                        control_deps: vec![],
                        shape: Some(shape),
                    });
                    graph.initializers.insert(zeros_id, zeros);

                    graph.nodes[grad_node_id].op =
                        std::rc::Rc::new(crate::engine::operation::IdentityOp);
                    graph.nodes[grad_node_id].inputs = vec![zeros_id];
                }
            }
        });
    }
}

pub fn handle_broadcast<B: Backend + 'static>(
    grad: &DTensor<B>,
    target: &DTensor<B>,
) -> DTensor<B> {
    let (g_shape, t_shape) = with_graph::<B, _, _>(|graph| {
        (
            graph.nodes[grad.id].shape.clone().unwrap(),
            graph.nodes[target.id].shape.clone().unwrap(),
        )
    });

    if g_shape == t_shape {
        return grad.clone();
    }

    let mut curr = grad.clone();
    let g_ndim = g_shape.len();
    let t_ndim = t_shape.len();

    // 1. Reduce extra leading dimensions
    if g_ndim > t_ndim {
        for _ in 0..(g_ndim - t_ndim) {
            curr = curr.sum(Some(0));
        }
    }

    // 2. Reduce dimensions that are 1 in target but >1 in grad
    // (This shouldn't happen for valid broadcast, but if target is 1, we must sum)
    // Actually standard broadcast rule: if target dim is 1, and grad dim is N, we sum.
    // We iterate over the aligned dimensions.
    // New g_ndim should be equal to t_ndim now (after step 1).
    let current_shape = with_graph::<B, _, _>(|graph| graph.nodes[curr.id].shape.clone().unwrap());

    // We need to iterate carefully.
    for i in 0..t_ndim {
        if t_shape[i] == 1 && current_shape[i] > 1 {
            curr = curr.sum_keepdims(Some(i));
        }
    }

    curr
}
