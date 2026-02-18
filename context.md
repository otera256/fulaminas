# Rust ML Framework: Unified Static Graph Architecture

## 1. プロジェクト概要
本プロジェクトは、Rustによる機械学習フレームワークの開発である。
最大の特長は、**「順伝搬・逆伝搬（自動微分）・パラメータ更新（Optimizer）」のすべてを単一の静的計算グラフ（Static Computation Graph）として表現し、1回の実行（Run）で学習の1サイクルを完結させる**点にある。
これにより、将来的にRNNの時間展開や、DFA（Direct Feedback Alignment）などの局所的な学習則、Optimizerのメタ学習などを統一的な枠組みで扱えるようにする。

## 2. アーキテクチャの基本原則
1. **Backend Abstraction (L1):** テンソル演算の実体は `Backend` トレイトで抽象化する。現在は `ndarray` を用いた CPUバックエンド (`NdArray`) が実装済み。将来的に `wgpu` などの追加を見据え、演算は常にジェネリクス `B: Backend` を伝播させる設計とする。
2. **Symbolic Tensor & Global Graph (L2):** ユーザーが操作する `Tensor<B>` はデータを持たず、計算グラフ上の `NodeId` のみを持つ。
   演算（`+`, `*`, `matmul` など）が呼ばれると、裏側で `thread_local!` な `GLOBAL_GRAPH_STORE` (TypeMapパターンによる `HashMap<TypeId, Box<dyn Any>>`) にアクセスし、バックエンドに対応した `GraphBuilder` にノードを追加する。
3. **Execution Engine (L3):** 構築されたグラフは `build()` によってコンパイルされる。ここでKahnのアルゴリズム（またはBFS/DFS）によるトポロジカルソートが行われ、実行順序 (`execution_order`) が確定した `Executor` が生成される。
4. **Data Ownership:**
   `Node<B>` の実行時データは `Option<B::Tensor>` として静的型付けで保持し、ダウンキャストのオーバーヘッドと実行時エラーを排除している。

## 3. 現在の実装状況（実装済みコードの要約）
ユーザーによって既に以下のコアコンポーネントが実装されている。

* **`backend/mod.rs` & `backend/ndarray.rs`:**
  `Backend` トレイトと `NdArray` の実装。バッチ処理を考慮したブロードキャスト対応の `matmul`（`rayon`の `par_azip!` を使用）や、乱数生成などが完了している。
* **`engine/node.rs`:**
  `NodeType` (Input, Parameter, Const, Operation, Assign) と `OpType` の定義。`Node<B>` 構造体。
* **`engine/tensor.rs`:**
  シンボリックな `Tensor<B>`。`new_input`, `new_parameter`, `new_const`, `op`, `assign` のヘルパー。`std::ops` を用いた演算子オーバーロード。
* **`engine/mod.rs`:**
  `GraphBuilder<B>` と `thread_local!` を用いた `with_graph` の実装（TypeMapパターン）。幅優先探索によるトポロジカルソートを含む `build()` 関数。
  *(※ 現在の `build()` は `Assign` ノードを無視する仕様になっている)*
* **`engine/executor.rs`:**
  `Executor<B>` と `run` メソッド。入力データをノードに割り当て、`execution_order` に従って順次計算 (`B::add`, `B::matmul` 等) を行い、出力を返す。
  *(※ 現在の `run` メソッドには `Assign` ノードの処理が含まれていない)*
* **`lib.rs` (テスト):**
  `test_add` にて、順伝搬のグラフ構築と実行が正しく行われることが確認されている。

## 4. 直近のマイルストーン (Phase 1 の完了)
現在は **Phase 1 (コア・グラフエンジンの実装)** の終盤である。以下の課題を解決し、変数の更新（Optimizerの基礎）が動作するようにする。

### Task 1: `Assign` ノードのトポロジカルソート対応
* `engine/mod.rs` の `build()` 内で `Assign` ノードをスキップしている部分を修正する。
* `Assign` は「副作用（変数の更新）」を持つため、ターゲットノードや入力ノードに正しく依存関係を設定し、実行順序リスト（`execution_order`）の適切な位置（通常は最後尾付近）に組み込まれるようにする。

### Task 2: `Executor` での `Assign` 実行対応
* `engine/executor.rs` の `run` メソッドにおいて、`NodeType::Assign` が来たときの処理を実装する。
* `Assign { target, depth }` の場合、入力ノードの計算結果を取得し、`self.nodes[target].data` を新しい値で上書きする。

### Task 3: Phase 1 完了テストの作成とパス
* `W_new = W + 1.0` とし、それを `Assign` で `W` に書き戻すような簡単な更新サイクルのテストコードを作成し、正常に `W` の内部状態が更新されることを確認する。

## 5. 次のフェーズ (Phase 2 & 3 への展望)
Phase 1 完了後は以下に進む。
* **Phase 2 (Autodiff):** `Tensor::backward()` を実装し、Chain Ruleに従って勾配計算ノード（`GradNode` などのオペレーション）を自動的にグラフに拡張・追加する機能を構築。
* **Phase 3 (Optimizer):** `SGD` などの Optimizer 構造体を作成し、計算された勾配ノードを利用して `Assign` ノードをグラフに追加する `step_graph()` のようなメソッドを実装する。