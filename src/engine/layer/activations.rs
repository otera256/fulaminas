// 活性化関数はすでにTensorのメソッドとして定義されているが、Layerとして定義することでジェネリクスに用いやすくなる

use crate::backend::Backend;
use crate::engine::layer::Layer;
use crate::engine::shape::Dynamic;
use crate::engine::tensor::Tensor;

type DTensor<B> = Tensor<B, Dynamic>;

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
    fn forward(&self, x: DTensor<B>) -> DTensor<B> {
        x.relu()
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
    fn forward(&self, x: DTensor<B>) -> DTensor<B> {
        x.sigmoid()
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
    fn forward(&self, x: DTensor<B>) -> DTensor<B> {
        x.tanh()
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
    fn forward(&self, x: DTensor<B>) -> DTensor<B> {
        x.softmax(Some(0))
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
    fn forward(&self, x: DTensor<B>) -> DTensor<B> {
        x.swish()
    }
}
