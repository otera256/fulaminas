// 活性化関数はすでにTensorのメソッドとして定義されているが、Layerとして定義することでジェネリクスに用いやすくなる

use crate::backend::Backend;
use crate::engine::layer::Layer;
use crate::engine::tensor::Tensor;

pub struct ReLU<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> ReLU<B> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Layer<B> for ReLU<B> {
    fn forward(&self, x: Tensor<B>) -> Tensor<B> {
        x.relu()
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

pub struct Sigmoid<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Sigmoid<B> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Layer<B> for Sigmoid<B> {
    fn forward(&self, x: Tensor<B>) -> Tensor<B> {
        x.sigmoid()
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

pub struct Tanh<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Tanh<B> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Layer<B> for Tanh<B> {
    fn forward(&self, x: Tensor<B>) -> Tensor<B> {
        x.tanh()
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

pub struct Softmax<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Softmax<B> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Layer<B> for Softmax<B> {
    fn forward(&self, x: Tensor<B>) -> Tensor<B> {
        x.softmax(Some(0))
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

pub struct Swish<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Swish<B> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Layer<B> for Swish<B> {
    fn forward(&self, x: Tensor<B>) -> Tensor<B> {
        x.swish()
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}
