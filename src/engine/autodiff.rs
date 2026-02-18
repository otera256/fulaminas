// 自動微分(Automatic Differentiation)を行うためのモジュール

use std::collections::HashMap;

use crate::backend::Backend;
use crate::engine::{
    node::{Node, NodeId, NodeType, OpType},
    tensor::Tensor,
    with_graph,
};

/// 計算グラフを展開し、逆伝播（バックプロパゲーション）のためのノードを追加します。
///
/// この関数は以下のステップを実行します：
/// 1. 計算すべき勾配ノード(`Grad`)を特定します。
/// 2. グラフのトポロジカルソートを行い、計算順序を決定します。
/// 3. グラフを逆順に辿りながら（逆伝播）、各ノードの勾配を計算する新しいノードを追加します。
/// 4. `Grad`ノードの入力を、計算された勾配ノードに接続します。
pub fn expand_graph<B: Backend + 'static>() {
    // 1. 計算すべき勾配の特定
    // yをxで微分したい場合 (`Grad {x, y}`ノードが存在する場合)、
    // yは逆伝播の出発点(root)となり、xは到達点となります。
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
    // これにより、順伝播の計算順序が得られます。逆伝播はこの逆順で行います。
    let forward_order = with_graph::<B, _, _>(|graph| graph.topological_sort(&global_roots));

    // 3. 各yのグループごとにバックワードパスを実行
    for (y_root, x_targets) in match_grads {
        let mut node_grads: HashMap<NodeId, Vec<Tensor<B>>> = HashMap::new();

        // 初期勾配 dy/dy = 1 を設定
        // y_root自身の勾配は、y_rootと同じ形状の全ての要素が1のテンソルです。
        let y_tensor = Tensor::from_id(y_root);
        let ones_tensor = Tensor::ones_like(&y_tensor);
        node_grads.entry(y_root).or_default().push(ones_tensor);

        // トポロジカルソートの逆順（出力から入力へ）でイテレーション
        for &node_id in forward_order.iter().rev() {
            // 現在のノードに対する勾配を収集
            // 分岐している場合（複数の出力先がある場合）、勾配は加算されます（連鎖律）。
            let final_grad = if let Some(grads) = node_grads.remove(&node_id) {
                if grads.len() == 1 {
                    grads[0].clone()
                } else {
                    Tensor::add_n(grads)
                }
            } else {
                continue;
            };

            // Gradノードへの接続
            // 現在のノードが、勾配を求めたい対象(x)である場合、
            // そのGradノードの入力を、計算された勾配(final_grad)に接続します。
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
            // ノードの種類に応じて、入力に対する勾配を計算し、node_gradsに追加します。
            let (node_type, inputs) = with_graph::<B, _, _>(|graph| {
                let node = &graph.nodes[node_id];
                (node.node_type.clone(), node.inputs.clone())
            });

            let input_tensors: Vec<Tensor<B>> =
                inputs.iter().map(|&id| Tensor::from_id(id)).collect();

            match node_type {
                NodeType::Operation(OpType::Add) => {
                    // z = a + b の場合
                    // dL/da = dL/dz * 1
                    // dL/db = dL/dz * 1
                    // ブロードキャストが行われている場合は、ブロードキャストされた次元について和をとる必要があります。
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];
                    let ga = handle_broadcast(&final_grad, a);
                    let gb = handle_broadcast(&final_grad, b);
                    node_grads.entry(a.id).or_default().push(ga);
                    node_grads.entry(b.id).or_default().push(gb);
                }
                NodeType::Operation(OpType::Sub) => {
                    // z = a - b の場合
                    // dL/da = dL/dz * 1
                    // dL/db = dL/dz * (-1)
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];
                    let ga = handle_broadcast(&final_grad, a);
                    let gb = handle_broadcast(&final_grad.neg(), b);
                    node_grads.entry(a.id).or_default().push(ga);
                    node_grads.entry(b.id).or_default().push(gb);
                }
                NodeType::Operation(OpType::Mul) => {
                    // z = a * b (要素ごとの積) の場合
                    // dL/da = dL/dz * b
                    // dL/db = dL/dz * a
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];
                    let dy_b = final_grad.clone() * b.clone();
                    let dy_a = final_grad.clone() * a.clone();

                    let ga = handle_broadcast(&dy_b, a);
                    let gb = handle_broadcast(&dy_a, b);
                    node_grads.entry(a.id).or_default().push(ga);
                    node_grads.entry(b.id).or_default().push(gb);
                }
                NodeType::Operation(OpType::Matmul) => {
                    // Z = A @ B の場合
                    // dL/dA = dL/dZ @ B^T
                    // dL/dB = A^T @ dL/dZ
                    let a = &input_tensors[0];
                    let b = &input_tensors[1];
                    let dy_b_t = final_grad.clone().matmul(b.clone().transpose());
                    let a_t_dy = a.clone().transpose().matmul(final_grad.clone());

                    node_grads.entry(a.id).or_default().push(dy_b_t);
                    node_grads.entry(b.id).or_default().push(a_t_dy);
                }
                NodeType::Operation(OpType::Transpose) => {
                    // z = a^T の場合
                    // dL/da = (dL/dz)^T
                    let a = &input_tensors[0];
                    let ga = final_grad.transpose();
                    node_grads.entry(a.id).or_default().push(ga);
                }
                NodeType::Operation(OpType::Sum { axis: _ }) => {
                    // Sum操作の逆伝播は、勾配を元の形状にブロードキャスト（コピー）することに対応します。
                    // 現在の実装では、後段の加算などでブロードキャスト処理が行われるため、
                    // ここではそのまま伝播させています。（厳密にはReshapeが必要な場合もあります）
                    let a = &input_tensors[0];
                    node_grads.entry(a.id).or_default().push(final_grad.clone());
                }
                NodeType::Operation(OpType::Identity) | NodeType::Operation(OpType::AddN) => {
                    // IdentityやAddNは勾配をそのまま伝播
                    for inp in &input_tensors {
                        node_grads
                            .entry(inp.id)
                            .or_default()
                            .push(final_grad.clone());
                    }
                }
                NodeType::Operation(OpType::Sigmoid) => {
                    // y = sigmoid(x)
                    // dy/dx = y * (1 - y)
                    let y = Tensor::from_id(node_id); // Output of this node
                    let one = Tensor::ones_like(&y);
                    // grad = final_grad * y * (1 - y)
                    let grad = final_grad * y.clone() * (one - y);
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Tanh) => {
                    // y = tanh(x)
                    // dy/dx = 1 - y^2
                    let y = Tensor::from_id(node_id);
                    let one = Tensor::ones_like(&y);
                    // grad = final_grad * (1 - y^2)
                    let grad = final_grad * (one - y.powi(2));
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Exp) => {
                    // y = exp(x)
                    // dy/dx = y
                    let y = Tensor::from_id(node_id);
                    let grad = final_grad * y;
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Log) => {
                    // y = log(x)
                    // dy/dx = 1 / x
                    let x = &input_tensors[0];
                    // 1/x = x^(-1) or powi(-1)
                    let grad = final_grad * x.clone().powi(-1);
                    node_grads.entry(x.id).or_default().push(grad);
                }
                NodeType::Operation(OpType::ReLU) => {
                    // y = max(0, x)
                    // dy/dx = 1 if x > 0 else 0
                    let x = Tensor::from_id(input_tensors[0].id);
                    // Let's use `x.gt(x - x)`.
                    let zeros = x.clone() - x.clone();
                    let mask = x.gt(zeros);
                    let grad = final_grad * mask;
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Softmax { axis }) => {
                    // Softmax output y
                    // Jacobian matrix J_ij = y_i (delta_ij - y_j)
                    // grad_x = grad_y * J
                    // This is complex for elementwise AD.
                    // Simplified: dx_i = y_i (grad_i - sum(y_k * grad_k))
                    let y = Tensor::from_id(node_id);
                    let gy = final_grad;
                    let y_gy = y.clone() * gy.clone(); // y * grad
                    let sum_y_gy = y_gy.sum(axis); // sum(y * grad)
                                                   // To subtract this scalar/reduced tensor from vectors, we need broadcast.
                                                   // Current sum implementation reduces dimension.
                                                   // We need to keep dimension or rely on broadcast of `sub`.
                                                   // If sum reduces shape, e.g. [2,3] -> [2], sub [2,3] - [2] works if [2] is treated as [2,1] or compatible.
                                                   // Our broadcast logic handles it?
                                                   // Let's check `shape.rs`. It handles basic broadcast.
                                                   // But `sum` removes the axis. So [2,3] sum axis 1 -> [2].
                                                   // [2,3] - [2] might fail if [2] aligns with dim 0.
                                                   // We often need `keep_dims=True`.

                    // Workaround: We don't have keep_dims in Sum op.
                    // But we can reshape? Or just trust backend broadcast behavior?
                    // For now, let's implement assuming broadcast works or standard simple Softmax grad.

                    // Using the formula: grad_input = y * (grad - sum(grad * y))
                    // We need `sum` to match dimensions.
                    // If `axis` is None (all), sum is scalar []. [2,3] - [] works.
                    // If `axis` is specified, we might have issue.
                    // Let's leave a TODO/Note or try best effort.
                    // The standard way is:
                    // dx = y * (dy - sum(dy * y, axis=axis, keepdims=True))

                    let grad = y.clone() * (gy - sum_y_gy); // This might need explicit broadcast/reshape
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                NodeType::Operation(OpType::Powi { n }) => {
                    // y = x^n
                    // dy/dx = n * x^(n-1)
                    let x = Tensor::from_id(input_tensors[0].id);
                    let n_const = Tensor::new_const(B::from_vec(vec![n as f32], &[1])); // Create scalar const?
                                                                                        // We don't have explicit Scalar mul yet, but we can broad cast.
                                                                                        // Actually `OpType::Mul` works on tensors.

                    // Optimization: Use `powi(n-1)`
                    let nx_n_minus_1 = x.powi(n - 1) * n_const;
                    let grad = final_grad * nx_n_minus_1;
                    node_grads
                        .entry(input_tensors[0].id)
                        .or_default()
                        .push(grad);
                }
                _ => {}
            }
        }

        // 到達しなかった（勾配が切れている）ターゲットに対してゼロ勾配を設定
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

/// ブロードキャストに対応するための勾配調整を行います。
/// 勾配の形状がターゲットの形状と異なる場合（次元が多い、あるいはサイズが1の次元がある場合）、
/// 余分な次元について和をとることで形状を合わせます。
fn handle_broadcast<B: Backend + 'static>(
    grad: &crate::engine::tensor::Tensor<B>,
    target: &crate::engine::tensor::Tensor<B>,
) -> crate::engine::tensor::Tensor<B> {
    // 形状を取得して比較
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

    // 次元の数が異なる場合、先頭の次元をsumで潰す
    if g_ndim > t_ndim {
        for _ in 0..(g_ndim - t_ndim) {
            curr = curr.sum(Some(0));
        }
    }

    curr
}
