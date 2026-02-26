use crate::backend::Backend;
use crate::engine::operation::{NoOp, Operation};
use std::rc::Rc;

/// 計算グラフ内のノードID（インデックス）
pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    Input,
    Const,
    LearnableParameter,
    OptimizerState,
    TargetMetric,
    /// Parameter update signal
    Feedback {
        target_param: NodeId,
    },
    None, // Intermediate calculation
}

#[derive(Clone)]
pub struct Node<B: Backend> {
    pub id: NodeId,
    pub role: Role,
    pub op: Rc<dyn Operation<B>>,
    pub inputs: Vec<NodeId>,
    pub control_deps: Vec<NodeId>,
    pub shape: Option<Vec<usize>>,
}

// We implement Debug manually because trait Object debug can be tricky depending on needs,
// or we can just derive Debug if we bounded Operation: Debug (which we did).
impl<B: Backend> std::fmt::Debug for Node<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("role", &self.role)
            .field("op", &self.op.name())
            .field("inputs", &self.inputs)
            .field("control_deps", &self.control_deps)
            .field("shape", &self.shape)
            .finish()
    }
}
