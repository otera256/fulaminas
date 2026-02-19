use super::dataset::Dataset;
use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use std::fs::{self, File};
use std::io::{self, Cursor, Read};
use std::path::Path;

const MNIST_URL_BASE: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";
const GR_FASHION_MNIST_URL_BASE: &str =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com";

pub enum MnistVariant {
    Mnist,
    FashionMnist,
}

pub struct Mnist {
    images: Vec<Vec<u8>>,
    labels: Vec<u8>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl Mnist {
    pub fn new(
        root: &str,
        train: bool,
        download: bool,
        variant: MnistVariant,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (image_filename, label_filename) = if train {
            ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
        } else {
            ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
        };

        let base_url = match variant {
            MnistVariant::Mnist => MNIST_URL_BASE,
            MnistVariant::FashionMnist => GR_FASHION_MNIST_URL_BASE,
        };

        let root_path = Path::new(root);
        if !root_path.exists() {
            fs::create_dir_all(root_path)?;
        }

        let image_path = root_path.join(image_filename);
        let label_path = root_path.join(label_filename);

        if download {
            if !image_path.exists() {
                println!("Downloading {}...", image_filename);
                download_file(&format!("{}/{}", base_url, image_filename), &image_path)?;
            }
            if !label_path.exists() {
                println!("Downloading {}...", label_filename);
                download_file(&format!("{}/{}", base_url, label_filename), &label_path)?;
            }
        }

        let images = read_idx3_file(&image_path)?;
        let labels = read_idx1_file(&label_path)?;

        assert_eq!(
            images.0.len(),
            labels.len(),
            "Image and label count mismatch"
        );

        Ok(Self {
            images: images.0, // data
            labels,
            num_rows: images.1,
            num_cols: images.2,
        })
    }
}

impl Dataset for Mnist {
    type Item = (Vec<f32>, u8); // Normalized pixels [0, 1], label

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn get(&self, index: usize) -> Self::Item {
        let img_data = &self.images[index];
        let label = self.labels[index];

        let normalized: Vec<f32> = img_data.iter().map(|&x| x as f32 / 255.0).collect();
        (normalized, label)
    }
}

fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let resp = ureq::get(url).call()?;
    let mut file = File::create(path)?;
    let mut reader = resp.into_reader();
    io::copy(&mut reader, &mut file)?;
    Ok(())
}

fn read_idx3_file(path: &Path) -> Result<(Vec<Vec<u8>>, usize, usize), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(file);
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer)?;

    let mut reader = Cursor::new(&buffer);

    let magic = reader.read_u32::<BigEndian>()?;
    if magic != 2051 {
        return Err("Invalid magic number for IDX3 file".into());
    }

    let count = reader.read_u32::<BigEndian>()? as usize;
    let rows = reader.read_u32::<BigEndian>()? as usize;
    let cols = reader.read_u32::<BigEndian>()? as usize;

    let mut images = Vec::with_capacity(count);
    let image_size = rows * cols;

    for _ in 0..count {
        let mut img_buf = vec![0u8; image_size];
        reader.read_exact(&mut img_buf)?;
        images.push(img_buf);
    }

    Ok((images, rows, cols))
}

fn read_idx1_file(path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(file);
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer)?;

    let mut reader = Cursor::new(&buffer);

    let magic = reader.read_u32::<BigEndian>()?;
    if magic != 2049 {
        return Err("Invalid magic number for IDX1 file".into());
    }

    let count = reader.read_u32::<BigEndian>()? as usize;
    let mut labels = vec![0u8; count];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}
