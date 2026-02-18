use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
};

use crate::{backend::Backend, engine::node::Node};

use self::{
    executor::Executor,
    node::{NodeId, NodeType},
};

pub mod executor;
pub mod layer;
pub mod model;
pub mod node;
pub mod shape;
pub mod tensor;

#[derive(Debug, Clone)]
pub struct GraphBuilder<B: Backend> {
    nodes: Vec<Node<B>>,
}

thread_local! {
    static GLOBAL_GRAPH_STORE: RefCell<HashMap<TypeId, Box<dyn Any>>> = RefCell::new(HashMap::new());
}

pub fn with_graph<B, F, R>(f: F) -> R
where
    B: Backend + 'static,
    F: FnOnce(&mut GraphBuilder<B>) -> R,
{
    GLOBAL_GRAPH_STORE.with(|store| {
        let mut map = store.borrow_mut();
        let type_id = TypeId::of::<B>();

        // グラフビルダーが存在しない場合は新規作成
        let graph_any = map
            .entry(type_id)
            .or_insert_with(|| Box::new(GraphBuilder::<B> { nodes: Vec::new() }));

        // グラフビルダーを取得して関数を実行
        let graph_builder = graph_any.downcast_mut::<GraphBuilder<B>>().unwrap();
        f(graph_builder)
    })
}

pub fn build<B: Backend + 'static>() -> Executor<B> {
    // コンパイルは１回しか呼ばれないはずなので、cloneのコストは許容できる
    with_graph::<B, _, _>(|graph| graph.clone().build())
}

impl<B: Backend> GraphBuilder<B> {
    fn build(self) -> Executor<B> {
        // Assignノードを全て取得し、それらをグラフの出力（ルート）として扱う
        let mut outputs = Vec::new();
        for node in &self.nodes {
            if let NodeType::Assign { .. } = node.node_type {
                outputs.push(node.id);
            }
        }

        // 幅優先探索により、出力ノードから逆順にノードを訪問する
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut adj = HashMap::<NodeId, Vec<NodeId>>::new();
        let mut in_degree = HashMap::<NodeId, usize>::new();
        for &output in &outputs {
            queue.push_back(output);
            visited.insert(output);
        }
        while let Some(node_id) = queue.pop_front() {
            in_degree.entry(node_id).or_insert(0);
            for &input in &self.nodes[node_id].inputs {
                if !visited.contains(&input) {
                    queue.push_back(input);
                    visited.insert(input);
                }
                adj.entry(input).or_insert_with(Vec::new).push(node_id);
                *in_degree.get_mut(&node_id).unwrap() += 1;
            }
        }
        let mut order = Vec::new();
        for (&id, &degree) in in_degree.iter() {
            if degree == 0 {
                queue.push_back(id);
            }
        }
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in adj.get(&u).unwrap_or(&Vec::new()) {
                *in_degree.get_mut(&v).unwrap() -= 1;
                if *in_degree.get(&v).unwrap() == 0 {
                    queue.push_back(v);
                }
            }
        }
        Executor::new(self.nodes, order)
    }
}
