use fulaminas::{
    backend::{Backend, ndarray::NdArray},
    data::{loader::DataLoader, mnist::Mnist},
    engine::tensor::Tensor,
};

fn main() {
    let mnist = Mnist::new(
        "./data/mnist",
        true,
        false, // Don't download, use local
        fulaminas::data::mnist::MnistVariant::Mnist,
    )
    .unwrap();
    let mnist_loader = DataLoader::new(&mnist, 1, true);
    for (i, batch) in mnist_loader.iter().enumerate().take(10) {
        let (vec, label) = batch[0].clone();
        // NdArray::from_vec returns NdArray's Tensor type (ArrayD<f32>)
        // Tensor::new_const expects B::Tensor. B is NdArray.
        // So this is correct.
        let image_data = <NdArray as Backend>::from_vec(vec, &[1, 28, 28]);
        let image_tensor = Tensor::<NdArray>::new_const(image_data);
        fulaminas::vis::save_image(&image_tensor, &format!("image_{}_label_{}.png", i, label))
            .unwrap();
    }
}
