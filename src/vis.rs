use crate::backend::Backend;
use crate::engine::shape::Shape;
use crate::engine::{tensor::Tensor, with_graph};
use image::{ImageBuffer, Luma, Rgb};

/// Saves a tensor as an image.
///
/// Supports 2D tensors (H, W) -> Grayscale.
/// Supports 3D tensors (C, H, W) -> Grayscale (if C=1) or RGB (if C=3).
/// Supports 4D tensors (1, C, H, W) -> Treated as (C, H, W).
/// Values are normalized to [0, 255].
pub fn save_image<B: Backend + 'static, S: Shape + Default>(
    tensor: &Tensor<B, S>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (data, shape) = with_graph::<B, _, _>(|graph| {
        let node = &graph.nodes[tensor.id()];
        let data = node.data.as_ref().expect("Tensor has no data").clone();
        let shape = node.shape.as_ref().expect("Tensor has no shape").clone();
        (data, shape)
    });

    let vec_data = B::to_vec(&data);
    let min_val = vec_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = vec_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let range = if (max_val - min_val).abs() < 1e-6 {
        1.0
    } else {
        max_val - min_val
    };

    let normalized: Vec<u8> = vec_data
        .iter()
        .map(|&x| ((x - min_val) / range * 255.0) as u8)
        .collect();

    // Squeeze batch dimension if it's 1
    // Explicitly annotate type
    let squeeze_shape: Vec<usize> = if shape.len() == 4 && shape[0] == 1 {
        shape[1..].to_vec()
    } else {
        shape.clone()
    };

    if squeeze_shape.len() == 2 {
        // (H, W) -> Grayscale
        let h = squeeze_shape[0] as u32;
        let w = squeeze_shape[1] as u32;
        let img: ImageBuffer<Luma<u8>, Vec<u8>> =
            ImageBuffer::from_vec(w, h, normalized).ok_or("Failed to create image buffer")?;
        img.save(path)?;
    } else if squeeze_shape.len() == 3 {
        // (C, H, W)
        let c = squeeze_shape[0];
        let h = squeeze_shape[1] as u32;
        let w = squeeze_shape[2] as u32;

        if c == 1 {
            let img: ImageBuffer<Luma<u8>, Vec<u8>> =
                ImageBuffer::from_vec(w, h, normalized).ok_or("Failed to create image buffer")?;
            img.save(path)?;
        } else if c == 3 {
            // (3, H, W) -> RGB
            // ImageBuffer expects RGBRGB... (interleaved)
            // Data is [[R..], [G..], [B..]] (planar)
            let mut rgb_data = Vec::with_capacity(normalized.len());
            let plane_size = (h * w) as usize;
            for i in 0..plane_size {
                rgb_data.push(normalized[i]); // R
                rgb_data.push(normalized[i + plane_size]); // G
                rgb_data.push(normalized[i + 2 * plane_size]); // B
            }
            let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_vec(w, h, rgb_data).ok_or("Failed to create image buffer")?;
            img.save(path)?;
        } else {
            return Err(format!("Unsupported channel count for image saving: {}", c).into());
        }
    } else {
        return Err(format!("Unsupported shape for image saving: {:?}", shape).into());
    }

    Ok(())
}
