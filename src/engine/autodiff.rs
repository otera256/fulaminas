// 自動微分(Automatic Differentiation)を行うためのモジュール

use std::collections::HashMap;

use crate::backend::Backend;
use crate::engine::{
    node::{Node, NodeId, NodeType, OpType},
    shape::Dynamic,
    tensor::Tensor,
    with_graph,
};

// AutoDiff内部では形状が動的に決定されるため、Dynamic Shapeを使用する
type DTensor<B> = Tensor<B, Dynamic>;

/// 計算グラフを展開し、逆伝播（バックプロパゲーション）のためのノードを追加します。
pub fn expand_graph<B: Backend + 'static>() {
    // 1. 計算すべき勾配の特定
    let (match_grads, global_roots) = with_graph::<B, _, _>(|graph| {
        let mut grads_to_process: HashMap<NodeId, Vec<(NodeId, NodeId)>> = HashMap::new();
        let mut global_roots = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            if let NodeType::Grad { x, y } = node.node_type {
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
                        graph.nodes[grad_node_id].node_type =
                            NodeType::Operation(crate::engine::node::OpType::Identity);
                        graph.nodes[grad_node_id].inputs = vec![final_grad.id];
                    }
                }
            });

            // バックプロパゲーション（入力への勾配伝播）
            let (node_type, inputs) = with_graph::<B, _, _>(|graph| {
                let node = &graph.nodes[node_id];
                (node.node_type.clone(), node.inputs.clone())
            });

            let input_tensors: Vec<DTensor<B>> =
                inputs.iter().map(|&id| DTensor::<B>::from_id(id)).collect();

            match node_type {
                NodeType::Operation(OpType::Add) => {
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];
                    let ga = handle_broadcast(&final_grad, a);
                    let gb = handle_broadcast(&final_grad, b);
                    node_grads.entry(a.id).or_default().push(ga);
                    node_grads.entry(b.id).or_default().push(gb);
                }
                NodeType::Operation(OpType::Sub) => {
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];
                    let ga = handle_broadcast(&final_grad, a);
                    let gb = handle_broadcast(&final_grad.neg(), b);
                    node_grads.entry(a.id).or_default().push(ga);
                    node_grads.entry(b.id).or_default().push(gb);
                }
                NodeType::Operation(OpType::Mul) => {
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];
                    let dy_b = final_grad.clone() * b.clone();
                    let dy_a = final_grad.clone() * a.clone();

                    let ga = handle_broadcast(&dy_b, a);
                    let gb = handle_broadcast(&dy_a, b);
                    node_grads.entry(a.id).or_default().push(ga);
                    node_grads.entry(b.id).or_default().push(gb);
                }
                NodeType::Operation(OpType::Div) => {
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];

                    let inv_b = b.clone().powi(-1);
                    let dy_da = final_grad.clone() * inv_b.clone();

                    let b_sq = b.clone().powi(2);
                    let neg_a_div_b_sq = a.clone().neg() * b_sq.powi(-1);
                    let dy_db = final_grad.clone() * neg_a_div_b_sq;

                    let ga = handle_broadcast(&dy_da, a);
                    let gb = handle_broadcast(&dy_db, b);
                    node_grads.entry(a.id).or_default().push(ga);
                    node_grads.entry(b.id).or_default().push(gb);
                }
                NodeType::Operation(OpType::Matmul) => {
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];
                    // Matmul gradients:
                    // dL/dA = dL/dZ @ B^T
                    // dL/dB = A^T @ dL/dZ
                    // We need generic matmul or dynamic one.
                    // Tensor::matmul is implemented for Rank2/Rank3.
                    // But here we have DTensor.
                    // We need to use `op` directly or implement matmul for DTensor?
                    // Tensor::matmul is not on `Tensor<B, S>` generally, only specialized impls.
                    // But `Tensor::op` is available.

                    let b_t = b.clone().transpose();
                    let dy_b_t = final_grad.clone().matmul(b_t);
                    // Wait, matmul method is not available on Dynamic? Eek.
                    // I need to add `matmul_dynamic` or make `matmul` generic over Rhs?
                    // Or just use `Tensor::op(Matmul, ...)` directly.
                    // Since `matmul` logic in backend/shape handles arbitrary shapes (generic broadcast matmul), it's fine to call OpType::Matmul.

                    let a_t = a.clone().transpose();
                    let a_t_dy = DTensor::op(OpType::Matmul, vec![&a_t, &final_grad]);

                    node_grads.entry(a.id).or_default().push(dy_b_t);
                    node_grads.entry(b.id).or_default().push(a_t_dy);
                }
                NodeType::Operation(OpType::Neg) => {
                    let a = &input_tensors[0];
                    let ga = final_grad.neg();
                    node_grads.entry(a.id).or_default().push(ga);
                }
                NodeType::Operation(OpType::Transpose) => {
                    let a = &input_tensors[0];
                    let ga = final_grad.transpose();
                    node_grads.entry(a.id).or_default().push(ga);
                }
                NodeType::Operation(OpType::Reshape { shape: _ }) => {
                    let a = &input_tensors[0];
                    let a_shape =
                        with_graph::<B, _, _>(|graph| graph.nodes[a.id].shape.clone().unwrap());
                    let ga = final_grad.reshape_dynamic(a_shape);
                    node_grads.entry(a.id).or_default().push(ga);
                }
                NodeType::Operation(OpType::Broadcast { shape: _ }) => {
                    // z = broadcast(a, shape)
                    // dL/da = sum(dL/dz) over broadcasted dimensions
                    let a = &input_tensors[0];
                    // handle_broadcast logic basically does the reduction
                    let ga = handle_broadcast(&final_grad, a);
                    node_grads.entry(a.id).or_default().push(ga);
                }
                NodeType::Operation(OpType::Sum { axis, keep_dims }) => {
                    let a = &input_tensors[0];
                    let mut grad = final_grad;

                    if !keep_dims {
                        if let Some(ax) = axis {
                            let mut g_shape = with_graph::<B, _, _>(|graph| {
                                graph.nodes[grad.id]
                                    .shape
                                    .clone()
                                    .expect("Grad has no shape")
                            });
                            g_shape.insert(ax, 1);
                            grad = grad.reshape_dynamic(g_shape);
                        }
                    }

                    let ones = DTensor::<B>::ones_like(a);
                    let expanded_grad = ones * grad; // Runtime broadcast

                    node_grads.entry(a.id).or_default().push(expanded_grad);
                }
                NodeType::Operation(OpType::Identity) | NodeType::Operation(OpType::AddN) => {
                    for inp in &input_tensors {
                        node_grads
                            .entry(inp.id)
                            .or_default()
                            .push(final_grad.clone());
                    }
                }
                NodeType::Operation(OpType::Sigmoid) => {
                    let y = DTensor::<B>::from_id(node_id);
                    let one = DTensor::<B>::ones_like(&y);
                    let grad = final_grad * y.clone() * (one - y);
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Tanh) => {
                    let y = DTensor::<B>::from_id(node_id);
                    let one = DTensor::<B>::ones_like(&y);
                    let grad = final_grad * (one - y.powi(2));
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Exp) => {
                    let y = DTensor::<B>::from_id(node_id);
                    let grad = final_grad * y;
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Log) => {
                    let x = &input_tensors[0];
                    let grad = final_grad * x.clone().powi(-1);
                    node_grads.entry(x.id).or_default().push(grad);
                }
                NodeType::Operation(OpType::ReLU) => {
                    let x = DTensor::<B>::from_id(input_tensors[0].id);
                    let zeros = x.clone() - x.clone();
                    let mask = x.gt(zeros);
                    let grad = final_grad * mask;
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Softmax { axis }) => {
                    let y = DTensor::<B>::from_id(node_id);
                    let gy = final_grad;
                    let y_gy = y.clone() * gy.clone();
                    let sum_y_gy = y_gy.sum_keepdims(axis);
                    let grad = y.clone() * (gy - sum_y_gy);
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Powi { n }) => {
                    let x = DTensor::<B>::from_id(input_tensors[0].id);
                    let n_const = DTensor::<B>::new_const(B::from_vec(vec![n as f32], &[1]));
                    let nx_n_minus_1 = x.powi(n - 1) * n_const; // Broadcast
                    let grad = final_grad * nx_n_minus_1;
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                _ => {}
            }
        }

        with_graph::<B, _, _>(|graph| {
            for &(x, grad_node_id) in &x_targets {
                if let NodeType::Grad { .. } = graph.nodes[grad_node_id].node_type {
                    let shape = graph.nodes[x].shape.clone().expect("Shape missing");
                    let zeros = B::zeros(&shape);

                    let zeros_id = graph.nodes.len();
                    graph.nodes.push(Node {
                        id: zeros_id,
                        node_type: NodeType::Const,
                        inputs: vec![],
                        data: Some(zeros),
                        shape: Some(shape),
                    });

                    graph.nodes[grad_node_id].node_type =
                        NodeType::Operation(crate::engine::node::OpType::Identity);
                    graph.nodes[grad_node_id].inputs = vec![zeros_id];
                }
            }
        });
    }
}

fn handle_broadcast<B: Backend + 'static>(grad: &DTensor<B>, target: &DTensor<B>) -> DTensor<B> {
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
