use crate::backend::Backend;
use crate::engine::layer::Layer;
use crate::engine::tensor::Tensor;

pub struct MLP<B: Backend> {
    layers: Vec<Box<dyn Layer<B>>>,
}

impl<B: Backend> MLP<B> {
    pub fn new(layers: Vec<Box<dyn Layer<B>>>) -> Self {
        Self { layers }
    }
}

impl<B: Backend> Layer<B> for MLP<B> {
    fn forward(&self, x: Tensor<B>) -> Tensor<B> {
        self.layers.iter().fold(x, |acc, layer| layer.forward(acc))
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
