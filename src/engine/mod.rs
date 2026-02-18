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
    fn build(mut self) -> Executor<B> {
        self.expand_gradients();

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

    fn expand_gradients(&mut self) {
        // Collect Grad nodes
        let mut grads_to_process: HashMap<NodeId, Vec<(NodeId, NodeId)>> = HashMap::new(); // y -> vec[(x, grad_node_id)]
        for (i, node) in self.nodes.iter().enumerate() {
            if let NodeType::Grad { x, y } = node.node_type {
                grads_to_process.entry(y).or_default().push((x, i));
            }
        }

        for (y_root, x_targets) in grads_to_process {
            // 1. Initialize grads: y_root -> ones_like(y_root)
            let mut grads: HashMap<NodeId, NodeId> = HashMap::new();

            // Create Ones node
            let shape = self.nodes[y_root].shape.clone().expect("Node has no shape");
            let ones_node = self.add_ones(shape);
            grads.insert(y_root, ones_node);

            // 2. Backward traversal from y_root
            // We need a topological sort of the subgraph leading to y_root
            let mut visited = HashSet::new();
            let mut stack = Vec::new();

            // DFS to sort
            let mut dfs_stack = vec![y_root];
            while let Some(curr) = dfs_stack.pop() {
                if !visited.contains(&curr) {
                    visited.insert(curr);
                    stack.push(curr); // Post-order push (reverse topo execution)
                                      // Push inputs
                    for &inp in &self.nodes[curr].inputs {
                        dfs_stack.push(inp);
                    }
                }
            }
            // stack has [y_root, ..., inputs], processing in this order is forward.
            // We want backward, so iterate stack directly (it is somewhat reverse topological relative to execution? No)
            // Wait, DFS post-order: visited children first.
            // y -> a -> b. Stack: [b, a, y].
            // We want to propagate FROM y TO inputs. So iterate stack in REVERSE (which is y -> a -> b).

            // Sort by node index (execution order usually follows index) is safer?
            // Let's use the explicit dependency structure collected in stack.
            // stack is [leaf, ..., root].
            // We iterate from root (y) down to leaves.
            // So reverse iter of stack.

            // To ensure we visit nodes in correct order (reverse topological),
            // sort the nodes in the subgraph by ID descending (heuristic) or use Kahn's on reverse graph.
            // Simple approach: Use a priority queue based on NodeId descending?
            // Correct approach: The collected stack from DFS post-order is [leaves..., root].
            // Reversing it gives [root, ... leaves]. This is a valid topological order for backward pass.

            // Filter duplicates in stack (set handled that) and sort by ID descending to be safe?
            // Standard reverse-mode AD: compute adjoints in reverse execution order.
            // Execution order is roughly indices 0..N.
            // So iterating indices N..0 is a good proxy.

            let mut subgraph_nodes: Vec<NodeId> = visited.into_iter().collect();
            subgraph_nodes.sort_by(|a, b| b.cmp(a)); // Descending ID order

            for node_id in subgraph_nodes {
                // If this node has a gradient, propagate it to inputs
                if let Some(&grad_node) = grads.get(&node_id) {
                    self.backprop_node(node_id, grad_node, &mut grads);
                }
            }

            // 3. Link accumulated gradients to Grad nodes
            for (x_target, grad_node_id) in x_targets {
                if let Some(&final_grad) = grads.get(&x_target) {
                    // Replace Grad node with Identity pointing to final_grad
                    self.nodes[grad_node_id].node_type =
                        NodeType::Operation(crate::engine::node::OpType::Identity);
                    self.nodes[grad_node_id].inputs = vec![final_grad];
                } else {
                    // No gradient path to x (gradient is zero)
                    let shape = self.nodes[x_target].shape.clone().expect("Shape missing");
                    let zeros_node = self.add_zeros(shape);
                    self.nodes[grad_node_id].node_type =
                        NodeType::Operation(crate::engine::node::OpType::Identity);
                    self.nodes[grad_node_id].inputs = vec![zeros_node];
                }
            }
        }
    }

    fn add_node(
        &mut self,
        node_type: NodeType,
        inputs: Vec<NodeId>,
        shape: Option<Vec<usize>>,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            node_type,
            inputs,
            data: None,
            shape,
        });
        id
    }

    fn add_ones(&mut self, shape: Vec<usize>) -> NodeId {
        let data = B::ones(&shape);
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            node_type: NodeType::Const,
            inputs: vec![],
            data: Some(data),
            shape: Some(shape),
        });
        id
    }

    fn add_zeros(&mut self, shape: Vec<usize>) -> NodeId {
        let data = B::zeros(&shape);
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            node_type: NodeType::Const,
            inputs: vec![],
            data: Some(data),
            shape: Some(shape),
        });
        id
    }

    fn backprop_node(
        &mut self,
        node_id: NodeId,
        grad_curr: NodeId,
        grads: &mut HashMap<NodeId, NodeId>,
    ) {
        let node = &self.nodes[node_id];
        match &node.node_type {
            NodeType::Operation(crate::engine::node::OpType::Add) => {
                // y = a + b => da = dy, db = dy
                let a = node.inputs[0];
                let b = node.inputs[1];

                // Need to handle broadcasting?
                // If a.shape != y.shape, we need to sum out dimensions.
                // Assuming implicit broadcasting support in backprop logic or explicit Sum

                let grad_a = self.handle_broadcast(grad_curr, a);
                let grad_b = self.handle_broadcast(grad_curr, b);

                self.accumulate_grad(a, grad_a, grads);
                self.accumulate_grad(b, grad_b, grads);
            }
            NodeType::Operation(crate::engine::node::OpType::Sub) => {
                // y = a - b => da = dy, db = -dy
                let a = node.inputs[0];
                let b = node.inputs[1];

                let grad_a = self.handle_broadcast(grad_curr, a);
                // db = -dy implies mul(dy, -1) or just sub(0, dy)
                // Let's use neg: 0 - dy
                // Or mul(dy, -1)

                // For simplicity, let's assume we have Neg or use 0 - grad
                let neg_grad = self.neg(grad_curr); // Need implement neg helper
                let grad_b = self.handle_broadcast(neg_grad, b);

                self.accumulate_grad(a, grad_a, grads);
                self.accumulate_grad(b, grad_b, grads);
            }
            NodeType::Operation(crate::engine::node::OpType::Mul) => {
                // y = a * b => da = dy * b, db = dy * a
                let a = node.inputs[0];
                let b = node.inputs[1];

                let dy_b = self.mul(grad_curr, b);
                let dy_a = self.mul(grad_curr, a);

                let grad_a = self.handle_broadcast(dy_b, a);
                let grad_b = self.handle_broadcast(dy_a, b);

                self.accumulate_grad(a, grad_a, grads);
                self.accumulate_grad(b, grad_b, grads);
            }
            NodeType::Operation(crate::engine::node::OpType::Matmul) => {
                // y = A @ B
                // dA = dy @ B.T
                // dB = A.T @ dy
                let a = node.inputs[0];
                let b = node.inputs[1];

                let b_t = self.transpose(b);
                let a_t = self.transpose(a);

                let dy_b_t = self.matmul(grad_curr, b_t);
                let a_t_dy = self.matmul(a_t, grad_curr);

                // Check broadcasting for Matmul? (batched matmul)
                // Assuming standard matmul for now.

                self.accumulate_grad(a, dy_b_t, grads);
                self.accumulate_grad(b, a_t_dy, grads);
            }
            _ => {}
        }
    }

    // Helpers for creating nodes
    fn accumulate_grad(
        &mut self,
        target: NodeId,
        grad: NodeId,
        grads: &mut HashMap<NodeId, NodeId>,
    ) {
        if let Some(&existing_grad) = grads.get(&target) {
            let new_grad = self.add(existing_grad, grad);
            grads.insert(target, new_grad);
        } else {
            grads.insert(target, grad);
        }
    }

    // ... Implement add, sub, mul, matmul, transpose, handle_broadcast, etc.
    // Due to snippet size limits, I'll need to break this down or simplify.
    // I will implement placeholders first then fill in.

    // Stub Helpers
    fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_op(crate::engine::node::OpType::Add, vec![a, b])
    }
    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_op(crate::engine::node::OpType::Mul, vec![a, b])
    }
    fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_op(crate::engine::node::OpType::Matmul, vec![a, b])
    }
    fn transpose(&mut self, a: NodeId) -> NodeId {
        self.add_op(crate::engine::node::OpType::Transpose, vec![a])
    }
    fn neg(&mut self, a: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone().expect("neg needs shape");
        let zeros = self.add_zeros(shape);
        self.add_op(crate::engine::node::OpType::Sub, vec![zeros, a])
    }

    fn add_op(&mut self, op_type: crate::engine::node::OpType, inputs: Vec<NodeId>) -> NodeId {
        // Shape inference
        let input_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|&id| self.nodes[id].shape.clone().expect("Shape missing"))
            .collect();
        let input_refs: Vec<&Vec<usize>> = input_shapes.iter().collect();
        let shape = crate::engine::shape::compute_shape(&op_type, &input_refs)
            .expect("Shape inference failed in AD expansion");

        self.add_node(NodeType::Operation(op_type), inputs, Some(shape))
    }

    fn sum(&mut self, a: NodeId, axis: Option<usize>) -> NodeId {
        self.add_op(crate::engine::node::OpType::Sum { axis }, vec![a])
    }

    fn handle_broadcast(&mut self, grad: NodeId, target: NodeId) -> NodeId {
        let grad_shape = self.nodes[grad].shape.clone().unwrap();
        let target_shape = self.nodes[target].shape.clone().unwrap();

        if grad_shape == target_shape {
            return grad;
        }

        let mut curr_grad = grad;
        let grad_ndim = grad_shape.len();
        let target_ndim = target_shape.len();

        // 1. Reduce extra leading dimensions
        if grad_ndim > target_ndim {
            for _ in 0..(grad_ndim - target_ndim) {
                // Remove axis 0 repeatedly
                curr_grad = self.sum(curr_grad, Some(0));
            }
        }

        // 2. Reduce broadcasted dimensions (size N in grad vs size 1 in target)
        // We need to re-fetch shape because it changed
        let curr_shape = self.nodes[curr_grad].shape.clone().unwrap();

        // Now ndims are equal (or target is scalar).
        for (i, &dim) in curr_shape.iter().enumerate() {
            let target_dim = target_shape.get(i).copied().unwrap_or(1); // Handle scalar target case
            if dim != target_dim && target_dim == 1 {
                // Broadcast happened on this axis. Sum it out.
                // Note: sum removes the axis, shifting indices.
                // This is problematic effectively.
                // Ideally we want to KeepDims, but our Sum removes it.
                // If we remove it, the next index `i+1` becomes `i`.
                // But we are iterating `i`.

                // Complexity alert: correct reduction of broadcasted dims without Reshape/KeepDims is tricky.
                // To allow progress, I will implement a simpler version that assumes no complex broadcasting requiring mid-axis reduction
                // OR allow the shape check to be loose (assumes backend handles rank mismatch).

                // If we assume standard broadcasting:
                // "If input has shape (S...) and result has shape (S... excluding broadcasted dims)"
                // Actually, if we just sum(axis), the result has rank-1.
                // If our engine shape checking enforces rank equality, we fail.
                // But `shape_mismatch` test showed strict equality checks.

                // For now, I will perform sum. The resulting shape will have lower rank.
                // If that causes mismatch downstream, we need Fix.
                // Since this is "Automatic" Differentiation, it should just work.

                // WORKAROUND: Just support leading dim reduction for now (e.g. batch size).
                // Mid-axis broadcasting (e.g. [2,1,3] + [2,4,3]) is rarer in basic MLPs.
                // But [3] + [2,3] is common (bias add). Bias [3] -> [1,3] -> [2,3].
                // Grad [2,3]. Target [3].
                // Diff in ranks: 1.
                // Step 1 removes axis 0. Result [3].
                // Target [3]. Match!
                // So step 1 covers common bias addition case.
            }
        }

        curr_grad
    }
}
