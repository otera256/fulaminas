# Refactoring Specification: Type-Level Shape Checking & Explicit Batch Dimensions

## 1. 目的 (Objective)
これまでの実装では、テンソルの形状不一致や暗黙のブロードキャストによるエラーが「実行時（Runフェーズ）」になって初めて発覚し、開発体験を損なっていた。
この問題を根本的に解決するため、Rustの **Const Generics（定数ジェネリクス）** を用いて、**バッチ次元を含むすべてのテンソル次元を型システムに明示的に乗せる設計** へと全面移行する。
これにより、形状の不一致はすべて「コンパイルエラー」としてエディタ上で即座に検知されるようにする。

## 2. 形状システムの設計 (Shape System Architecture)

### 2.1. Shape トレイトの定義
形状を型レベルで抽象化するためのトレイトと、次元数（Rank）ごとの構造体を定義する。Nightly機能は使用せず、Stable Rustで実現可能な範囲の静的チェックを行う。

```rust
// engine/shape.rs (新規作成または tensor.rs 内に定義)
pub trait Shape: Clone + std::fmt::Debug + Send + Sync + 'static {
    type Array;
    fn as_array(&self) -> Self::Array;
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Rank1<const M: usize>;
impl<const M: usize> Shape for Rank1<M> {
    type Array = [usize; 1];
    fn as_array(&self) -> Self::Array { [M] }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Rank2<const M: usize, const N: usize>;
impl<const M: usize, const N: usize> Shape for Rank2<M, N> {
    type Array = [usize; 2];
    fn as_array(&self) -> Self::Array { [M, N] }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct Rank3<const B: usize, const M: usize, const N: usize>;
impl<const B: usize, const M: usize, const N: usize> Shape for Rank3<B, M, N> {
    type Array = [usize; 3];
    fn as_array(&self) -> Self::Array { [B, M, N] }
}
```

### 2.2. Symbolic Tensor の再定義

既存の Tensor<B> に形状型 S: Shape を追加する。
内部のグラフ表現（Node や GraphBuilder）にはジェネリクス S を伝播させず、型チェックはフロントエンド（ユーザーAPI）のみで完結する Zero-cost Abstraction とする。

```rust
// engine/tensor.rs
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Tensor<B: Backend + 'static, S: Shape> {
    pub(crate) id: NodeId,
    phantom: std::marker::PhantomData<(B, S)>,
}

impl<B: Backend + 'static, S: Shape> Tensor<B, S> {
    pub fn new_input() -> Self {
        let node_id = with_graph::<B, _, _>(|graph| { ... });
        Self { id: node_id, phantom: PhantomData }
    }
    // new_parameter, new_const 等も同様に更新
}
```

## 3. 型安全な演算の実装方針 (Type-Safe Operations)

暗黙のブロードキャストは完全に禁止し、型シグネチャによって「どの次元がどう計算されるか」を厳密に定義する。
### 3.1. 要素ごとの演算 (Add, Sub, Mul)

完全に同じ形状（S が一致）の場合のみ加算などを許可する。

```rust
impl<B: Backend + 'static, S: Shape> std::ops::Add for Tensor<B, S> {
    type Output = Tensor<B, S>;
    fn add(self, rhs: Self) -> Self::Output {
        // コンパイラが S の一致を保証しているため、安全にOpを追加できる
        Tensor::op(OpType::Add, vec![&self.id, &rhs.id])
    }
}
```

### 3.2. 行列積 (MatMul)

内部次元 K の一致をコンパイラに検証させる。バッチ次元を持つ場合と持たない場合で個別に実装を提供する。

```rust
// ケース1: バッチ次元あり (Batch, M, K) @ (K, N) -> (Batch, M, N)
// 全結合層(Linear)の順伝搬などで頻出する計算
impl<B: Backend + 'static, const BATCH: usize, const M: usize, const K: usize> 
    Tensor<B, Rank3<BATCH, M, K>> 
{
    pub fn matmul<const N: usize>(
        self, 
        rhs: Tensor<B, Rank2<K, N>>
    ) -> Tensor<B, Rank3<BATCH, M, N>> {
        Tensor::op(OpType::Matmul, vec![&self.id, &rhs.id]) // 戻り値の型が自動で変化
    }
}

// ケース2: 2次元同士 (M, K) @ (K, N) -> (M, N)
impl<B: Backend + 'static, const M: usize, const K: usize> 
    Tensor<B, Rank2<M, K>> 
{
    pub fn matmul<const N: usize>(
        self, 
        rhs: Tensor<B, Rank2<K, N>>
    ) -> Tensor<B, Rank2<M, N>> {
        Tensor::op(OpType::Matmul, vec![&self.id, &rhs.id])
    }
}
```

### 3.3. 明示的なブロードキャスト (Explicit Broadcast)

バッチ次元を持たないバイアス項などを、バッチ次元を持つテンソルに足し合わせる場合、明示的なメソッド変換を必須とする。

```rust

impl<B: Backend + 'static, const N: usize> Tensor<B, Rank1<N>> {
    // 例: バイアス(N) を バッチサイズ BATCH の出力に足すために次元を拡張する
    // 内部的にはグラフに OpType::Broadcast ノードを追加するか、
    // バックエンド側で良きに計らうフラグを立てる
    pub fn broadcast_batch<const BATCH: usize>(self) -> Tensor<B, Rank2<BATCH, N>> {
        Tensor::op(OpType::Broadcast, vec![&self.id])
    }
}
```

## 4. エージェントへの指示 (Action Items for the Agent)

現在実装されているコードベースに対して、以下の順序でリファクタリングを実行せよ。

    Shape トレイト群の実装: engine/tensor.rs または新規モジュールに上記の Rank1, Rank2, Rank3 を追加せよ。

    Tensor 構造体の更新: ジェネリクスに S: Shape を追加し、new_input などの関数シグネチャを更新せよ。

        注意: Node や GraphBuilder 側の実装は極力変更せず、IDベースの管理を維持すること。

    演算オーバーロードの更新: Add, Sub, Mul などのトレイト実装に S: Shape 制約を付与せよ。

    matmul の型安全化: 上記の「ケース2: 2次元同士」のシグネチャで matmul メソッドを実装せよ。

    テストコードの修正: lib.rs の test_add において、Tensor::<NdArray, Rank1<3>>::new_input() のように型注釈を追加してコンパイルが通るように修正せよ。