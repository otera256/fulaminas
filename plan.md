struct Model_1<B: Backend> {
    linear1: Linear<B>,
    activation1: ReLU<B>,
    linear2: Linear<B>,
}

impl<B: Backend> Layer<B> for Model_1<B> {
    fn forward(&self, input: Tensor<B>) -> Tensor<B> {
        // BがAutodiff<B>の場合はこのとき順伝搬の計算グラフが構築される
        let x = self.linear1.forward(input);
        let x = self.activation1.forward(x);
        self.linear2.forward(x)
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        // モデルのパラメータを返す
        let mut params = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }
    fn from_parameters(parameters: Vec<Tensor<B>>) -> Self {
        // parametersからモデルを再構築する
        let linear1_params = parameters[0..2].to_vec(); // 例えば、重みとバイアス
        let linear2_params = parameters[2..4].to_vec(); // 例えば、重みとバイアス
        Self {
            linear1: Linear::from_parameters(linear1_params),
            activation1: ReLU::new(),
            linear2: Linear::from_parameters(linear2_params),
        }
    }
}

enum NodeType {
    Input,
    Parameter,
    Operation,
}

type NodeId = usize;

struct Node {
    id: NodeId,
    node_type: NodeType,
    value: Option<Tensor>, // ノードの値（入力やパラメータの場合は初期値、演算ノードの場合は計算結果）
}

struct ComputeGraph {
    nodes: Vec<Node>, // 計算グラフのノード
    edges: Vec<(NodeId, NodeId)>, // ノード間の依存関係（エッジ）
    sorted_nodes: Vec<NodeId>, // トポロジカルソートされたノードの順序
}

struct TrainingModel<M: Layer<B>, B: AutodiffBackend, O: OptimizerConfig> {
    compute_graph: ComputeGraph, // 順伝搬から逆伝搬まで含めた学習1サイクルの計算グラフ
}

impl<M: Layer<B>, B: AutodiffBackend, O: OptimizerConfig> TrainingModel<M, B, O> {
    fn new(model: M, optimizer_config: O) -> Self {
        // モデルの全パラメータを取得
        let parameters = model.parameters();

        // 学習用の計算グラフは学習するパラメータ、モデル中のアクティベーション、パラメータの勾配のノードを含む
        let mut compute_graph = ComputeGraph::new();

        // 順伝搬の計算グラフを構築
        let input = B::Tensor::new_input(); // 入力ノード
        let output = model.forward(input.clone()); // 順伝搬

        let label_input = model.new_label_input(); // ラベル入力ノード

        // 逆伝搬の計算グラフを構築
        let loss = MSELoss::new(output, label_input); // 例えば、平均二乗誤差の損失ノード
        loss.backward(); // 逆伝搬

        // どうやってグラフとして情報を集めてくるのかが課題。モデルのforwardで計算グラフが構築されるので、そこからノードとエッジを集めてくる必要がある。
        for param in &parameters {
            compute_graph.extend_from(param); // パラメータノードとその勾配ノードを計算グラフに追加
            // オプティマイザー用の計算ノードも追加
            compute_graph.add_optimizer_node(optimizer_config, param, param.grad()); // 例えば、SGDの更新ノードを追加
        }

        Self {
            compute_graph,
        }
    } 
    fn step(&mut self, input: Vec<Tensor<B>>) {
        // 順伝搬と逆伝搬とパラメータ更新を1サイクルで実行するのがこのライブラリで一番実現したいところ。将来的にはオプティマイザーのための特別な計算ノードとRNN用の計算ノードを同じ枠組みで扱いたい
        self.compute_graph.run(input); // 順伝搬と逆伝搬を実行して、パラメータの更新まで行う
    }
}

fn main() {
    // モデルの定義
    let model = Model_1 {
        linear1: Linear::new(28 * 28, 128),
        activation1: ReLU::new(),
        linear2: Linear::new(128, 10),
    };

    // 学習用のモデルを作成
    let mut training_model = TrainingModel::new(model, SGDConfig { learning_rate: 0.01 });

    let dataset = load_mnist(); // MNISTデータセットをロード


    // 学習ループ
    for epoch in 0..100 {
        let (inputs, labels) = dataset.next_batch(64); // バッチサイズ64でデータを取得
        training_model.step(vec![inputs, labels]); // 学習ステップを実行
    }
}
