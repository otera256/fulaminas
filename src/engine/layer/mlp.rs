use crate::backend::Backend;
use crate::engine::layer::Layer;
use crate::engine::shape::Dynamic;
use crate::engine::tensor::Tensor;

type DTensor<B> = Tensor<B, Dynamic>;

pub struct MLP<B: Backend> {
    layers: Vec<Box<dyn Layer<B>>>,
}

impl<B: Backend> MLP<B> {
    pub fn new(layers: Vec<Box<dyn Layer<B>>>) -> Self {
        Self { layers }
    }
}

impl<B: Backend> Layer<B> for MLP<B> {
    fn forward(&self, x: DTensor<B>) -> DTensor<B> {
        self.layers.iter().fold(x, |acc, layer| layer.forward(acc))
    }

    fn parameters(&self) -> Vec<DTensor<B>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
