use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
};

use crate::{backend::Backend, engine::node::Node};

use self::{executor::Executor, node::NodeId};

pub mod autodiff;
pub mod executor;
pub mod layer;
pub mod loss;
pub mod metric;
pub mod model;
pub mod node;
pub mod operation;
pub mod optimizer;
pub mod shape;
pub mod tensor;

#[derive(Debug, Clone)]
pub struct GraphBuilder<B: Backend> {
    pub nodes: Vec<Node<B>>,
    pub initializers: HashMap<NodeId, B::Tensor>,
}

thread_local! {
    /// グローバルなグラフストア。
    /// 各スレッドごとに、バックエンドの型(`TypeId`)をキーとして`GraphBuilder`を保持します。
    /// `with_graph`関数を通じてアクセスされます。
    static GLOBAL_GRAPH_STORE: RefCell<HashMap<TypeId, Box<dyn Any>>> = RefCell::new(HashMap::new());
}

/// グローバルな`GraphBuilder`にアクセスするためのヘルパー関数。
///
/// クロージャ`f`の中で`GraphBuilder`への可変参照を受け取り、ノードの追加や参照を行います。
/// スレッドローカルストレージを使用しているため、スレッドセーフです。
pub fn with_graph<B, F, R>(f: F) -> R
where
    B: Backend + 'static,
    F: FnOnce(&mut GraphBuilder<B>) -> R,
{
    GLOBAL_GRAPH_STORE.with(|store| {
        let mut map = store.borrow_mut();
        let type_id = TypeId::of::<B>();

        // グラフビルダーが存在しない場合は新規作成
        let graph_any = map.entry(type_id).or_insert_with(|| {
            Box::new(GraphBuilder::<B> {
                nodes: Vec::new(),
                initializers: HashMap::new(),
            })
        });

        // グラフビルダーを取得して関数を実行
        let graph_builder = graph_any.downcast_mut::<GraphBuilder<B>>().unwrap();
        f(graph_builder)
    })
}

/// 計算グラフを構築し、実行可能な`Executor`を返します。
///
/// この関数は以下の処理を行います：
/// 1. `autodiff::expand_graph`を呼び出し、自動微分に必要な勾配計算ノードを展開します。
/// 2. 現在のグラフの状態をクローンし、`Executor`を生成します。
pub fn build<B: Backend + 'static>() -> Executor<B> {
    // 勾配展開をここで行う (Tensor演算を使うため)
    autodiff::expand_graph::<B>();

    // コンパイルは１回しか呼ばれないはずなので、cloneのコストは許容できる
    with_graph::<B, _, _>(|graph| graph.clone().build())
}

impl<B: Backend> GraphBuilder<B> {
    /// `Executor`を生成します。
    ///
    /// グラフ内の`Assign`ノードを出力（ルート）と見なし、
    /// それらの値を計算するために必要なノードの実行順序（トポロジカルソート）を決定します。
    fn build(self) -> Executor<B> {
        let mut training_roots = Vec::new();
        let mut inference_roots = Vec::new();

        for node in &self.nodes {
            // Assign nodes act as roots (outputs to be computed)
            if node.op.name().starts_with("Assign") {
                training_roots.push(node.id);

                // For inference, exclude parameter/optimizer updates
                // inputs[1] is the target node
                if node.inputs.len() > 1 {
                    let target_id = node.inputs[1];
                    let target_role = &self.nodes[target_id].role;
                    match target_role {
                        node::Role::LearnableParameter | node::Role::OptimizerState => {
                            // Exclude from inference
                        }
                        _ => {
                            // Include in inference (e.g. assigning to loss container, accuracy, etc.)
                            inference_roots.push(node.id);
                        }
                    }
                }
            } else if let node::Role::TargetMetric = node.role {
                // Also ensure TargetMetric nodes are computed if they exist
                training_roots.push(node.id);
                inference_roots.push(node.id);
            }
        }

        let training_plan = self.topological_sort(&training_roots);
        let inference_plan = self.topological_sort(&inference_roots);

        Executor::new(self.nodes, self.initializers, training_plan, inference_plan)
    }

    /// 指定されたルートノード群から辿れるグラフのトポロジカルソート（実行順序）を返します。
    /// Kahnのアルゴリズムを使用しています。
    fn topological_sort(&self, roots: &[NodeId]) -> Vec<NodeId> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut adj = HashMap::<NodeId, Vec<NodeId>>::new();
        let mut in_degree = HashMap::<NodeId, usize>::new();

        for &root in roots {
            queue.push_back(root);
            visited.insert(root);
        }

        // 1. サブグラフの探索 (出力から入力への逆BFS)
        // 計算に必要なノードのみを抽出します。
        while let Some(node_id) = queue.pop_front() {
            in_degree.entry(node_id).or_insert(0);

            for &input in &self.nodes[node_id].inputs {
                if !visited.contains(&input) {
                    queue.push_back(input);
                    visited.insert(input);
                }

                // 依存関係のエッジを作成: input -> node_id
                adj.entry(input).or_insert_with(Vec::new).push(node_id);
                *in_degree.get_mut(&node_id).unwrap() += 1;
            }
        }

        // 2. Kahnのアルゴリズムによるトポロジカルソート
        // 入力次数が0のノードから順にキューに追加していきます。
        let mut queue = VecDeque::new();
        for (&id, &degree) in in_degree.iter() {
            if degree == 0 {
                queue.push_back(id);
            }
        }

        let mut order = Vec::new();
        while let Some(u) = queue.pop_front() {
            order.push(u);

            if let Some(neighbors) = adj.get(&u) {
                for &v in neighbors {
                    if let Some(d) = in_degree.get_mut(&v) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push_back(v);
                        }
                    }
                }
            }
        }

        order
    }
}
