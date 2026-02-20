# Refactoring Specification v2: Role-Based Stateful Graph & Explicit Dataflow

## 1. 目的 (Objective)

現在の計算グラフエンジンは正常に動作しているが、「ノードの役割（意味）と操作の混同」「実行時のメモリと静的な設計図の結合」「暗黙のブロードキャストによる逆伝播の複雑化」により、拡張性が限界に達している。
本リファクタリングでは、これらを分離・整理し、**「DFAやPredictive Codingなどの局所学習則を容易に実装でき、Adamのような複雑な状態更新の順序を数学的に保証できる、完全な静的グラフコンパイラ」**へとアーキテクチャを刷新する。

## 2. コア・アーキテクチャの変更点

### 2.1. グラフ（設計図）とメモリ（状態）の完全分離

課題: Node が data: Option<B::Tensor> を持っているため、グラフ構築と実行の境界が曖昧になり、コンパイル時のグラフ全体のクローンなど不要なオーバーヘッドが生じている。

解決策: Node から data を削除。実行時のテンソル実体は Executor 内部の HashMap<NodeId, B::Tensor> (Memory) だけで管理する。

### 2.2. 「役割 (Role)」と「操作 (Op)」の分離と相対化

課題: NodeType::Grad が操作と役割を混同しており、自動微分のハードコードに繋がっていた。

解決策: ノードがグラフ上で果たす「役割（Role）」と、実行エンジンが処理する「操作（Op）」を分ける。勾配は特定の役割 Feedback として相対化し、Optimizerは Feedback ロールを持つノードを探して更新式を組む。

### 2.3. 暗黙のブロードキャストの禁止（明示化）

課題: autodiff.rs の handle_broadcast が非常に複雑でバグの温床。

解決策: グラフ上での暗黙の形状拡張を禁止する。フロントエンドの + 演算などで形状が異なる場合は、必ずグラフに Broadcast ノードを自動挿入してから Add ノードを作成する。これにより Add の逆伝播は常に純粋な1:1のコピーとなり、Broadcast の逆伝播は常に Sum になる。

### 2.4. 副作用の順序保証 (Control Dependencies)

課題: Adamの m, v の更新が p の更新より先に行われる保証がない。

解決策: Assign オペレーションが「更新後の新しい値」を出力するように仕様変更し、それを後続の計算に渡す（データフローによる順序保証）。さらに、データ依存がない場合のために control_deps (制御依存) を Node に追加する。

### 2.5. マルチ実行プラン (Execution Plans)

課題: 推論時と学習時で Assign などの実行有無を切り替えたい。

解決策: Executor は単一の順序ではなく、inference_plan と training_plan など、フェーズごとの実行順序リストを持つ。

## 3. データ構造の再定義 (Data Structures)

### 3.1. Role と Node の定義 (engine/node.rs)

```rust
use crate::backend::Backend;

pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    Input,
    Const,
    LearnableParameter,
    OptimizerState,
    TargetMetric, // Lossなど
    /// パラメータ更新のためのシグナル。Backpropなら勾配、DFAなら射影誤差。
    Feedback { target_param: NodeId }, 
    None, // 単なる中間計算
}

#[derive(Debug, Clone)]
pub struct Node<B: Backend> {
    pub id: NodeId,
    pub role: Role,
    pub op: OpType, // ※将来的にBox<dyn Operation<B>>へ移行するが、今回はenumを維持
    
    pub inputs: Vec<NodeId>,
    pub control_deps: Vec<NodeId>, // このノードの実行前に完了しているべきNodeId
    
    pub shape: Option<Vec<usize>>,
    // data: Option<B::Tensor> は削除！
}
```

### 3.2. Executor の定義 (engine/executor.rs)

```rust
use std::collections::HashMap;

pub struct Executor<B: Backend> {
    pub nodes: Vec<Node<B>>,
    pub training_plan: Vec<NodeId>,
    pub inference_plan: Vec<NodeId>,
    pub memory: HashMap<NodeId, B::Tensor>, // 状態とキャッシュ
}

impl<B: Backend> Executor<B> {
    pub fn step_train(&mut self, inputs: Vec<(NodeId, B::Tensor)>) { ... }
    pub fn step_inference(&mut self, inputs: Vec<(NodeId, B::Tensor)>) { ... }
}
```

## 4. コンパイラ・パスの設計 (Compiler Passes)

mod.rs の build() 関数は、単なるトポロジカルソートではなく、「中間表現（IR）の最適化と拡張」 のパイプラインとなる。

```rust
pub fn build<B: Backend + 'static>() -> Executor<B> {
    // 1. Backprop Strategy Pass
    // TargetMetric を起点に、LearnableParameter までの逆伝播グラフを構築し、
    // 生成されたノードに Role::Feedback { target_param } を付与する。
    autodiff::apply_backprop_pass::<B>();

    // 2. Optimizer Strategy Pass
    // 各 LearnableParameter に対して、紐づく Role::Feedback を探し、
    // Assign ノードを用いた更新グラフ (m, v, p 等) を追加する。
    optimizer::apply_optimizer_pass::<B>();

    // 3. 実行計画の作成とExecutorの生成
    with_graph::<B, _, _>(|graph| {
        // ... training_plan と inference_plan をそれぞれトポロジカルソートで生成
        Executor::new(graph.nodes.clone(), training_plan, inference_plan)
    })
}
```

## 5. エージェントへの指示 (Action Items)

本リファクタリングは極めて広範囲に及ぶため、以下の順序で段階的に実装せよ。

Phase 1: グラフとメモリの分離 (Data Decoupling)

node.rs: NodeType のまま作業してよいが、Node 構造体から data: Option<B::Tensor> を削除せよ。

executor.rs: Executor 構造体に memory: HashMap<NodeId, B::Tensor> を追加せよ。

executor.rs: run メソッドのシグネチャと実装を変更し、node.data ではなく self.memory に対して読み書きを行うようにせよ。

Phase 2: Role の導入と Assign の改良

node.rs: NodeType を廃止し、Role Enum と OpType Enum に分割せよ。Node は role, op, inputs, control_deps を持つようにせよ。

tensor.rs: 各 new_xxx 関数で適切な Role を設定するようにせよ。

tensor.rs: Tensor::assign(target, value) を更新し、戻り値として「更新後の値を表すTensor（実態はAssignノードのID）」を返すようにせよ。これによりデータフローによる順序保証が可能になる。

Phase 3: 明示的ブロードキャストの強制 (Explicit Broadcast)

tensor.rs: Add, Sub, Mul, Div の演算子オーバーロードにおいて、両者の shape が異なる場合、自動的に compute_broadcast_shape でターゲット形状を求め、双方に対して Broadcast ノードを挿入してから実演算の OpType を追加するようにせよ。

autodiff.rs: handle_broadcast 関数を完全に削除せよ。Add や Mul の逆伝播は形状同一を前提としたシンプルなものに書き換え、Broadcast の逆伝播として新しく Sum (次元を潰す処理) を実装せよ。

Phase 4: Strategy Pass の実装

autodiff.rs: expand_graph を apply_backprop_pass に改名し、NodeType::Grad を探す処理から Role::TargetMetric を探す処理に変更せよ。生成した勾配ノードの末端には Role::Feedback を付与せよ。

optimizer.rs: Adam などの step メソッドを改修し、グラフ内から Role::Feedback を持つノードを探して更新グラフを構築するようにせよ。また、$m$ と $v$ の更新(Assign)の戻り値を使って $p$ の更新を行うことで、順序を数学的に保証せよ。