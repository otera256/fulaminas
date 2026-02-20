// use crate::backend::Backend; // Unused now

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

#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    Transpose,
    Reshape {
        shape: Vec<usize>,
    },
    Broadcast {
        shape: Vec<usize>,
    },
    Sum {
        axis: Option<usize>,
        keep_dims: bool,
    },
    Neg,
    Identity,
    AddN,
    OnesLike,

    // Activations
    Sigmoid,
    Tanh,
    ReLU,
    Softmax {
        axis: Option<usize>,
    },
    Exp,
    Log,
    Powi {
        n: i32,
    },

    // Comparison / Logic
    Gt,
    Eq,

    // Other
    Sqrt,
    ArgMax {
        axis: usize,
    },

    // New Control Ops
    Assign {
        depth: usize,
    }, // depth allows scheduling prioritization
    NoOp, // For Input, Parameter, Const
    Grad {
        x: NodeId,
        y: NodeId,
    }, // Legacy placeholder for autodiff expansion
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub role: Role,
    pub op: OpType,
    pub inputs: Vec<NodeId>,
    pub control_deps: Vec<NodeId>,
    pub shape: Option<Vec<usize>>,
}
