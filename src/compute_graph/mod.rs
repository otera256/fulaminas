use std::{any::{Any, TypeId}, cell::RefCell, collections::HashMap};

use crate::{backend::Backend, compute_graph::node::Node};

pub mod node;
pub mod tensor;

pub struct GraphBuilder<B: Backend> {
    nodes: Vec<Node<B>>
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
        let graph_any = map.entry(type_id).or_insert_with(|| 
            Box::new(GraphBuilder::<B> { nodes: Vec::new() })
        );

        // グラフビルダーを取得して関数を実行
        let graph_builder = graph_any.downcast_mut::<GraphBuilder<B>>().unwrap();
        f(graph_builder)
    })
}