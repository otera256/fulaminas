use super::dataset::Dataset;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
        }
    }

    pub fn iter(&self) -> DataLoaderIterator<'a, D> {
        let mut indices = self.indices.clone();
        if self.shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }

        DataLoaderIterator {
            dataset: self.dataset,
            indices,
            batch_size: self.batch_size,
            current_idx: 0,
        }
    }
}

pub struct DataLoaderIterator<'a, D: Dataset> {
    dataset: &'a D,
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
}

impl<'a, D: Dataset> Iterator for DataLoaderIterator<'a, D> {
    type Item = Vec<D::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let batch: Vec<D::Item> = batch_indices.iter().map(|&i| self.dataset.get(i)).collect();

        self.current_idx += self.batch_size;
        Some(batch)
    }
}
