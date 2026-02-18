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
