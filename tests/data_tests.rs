use fulaminas::data::dataset::Dataset;
use fulaminas::data::loader::DataLoader;

struct DummyDataset {
    data: Vec<i32>,
}

impl Dataset for DummyDataset {
    type Item = i32;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Self::Item {
        self.data[index]
    }
}

#[test]
fn test_dataloader_batching() {
    let data: Vec<i32> = (0..10).collect();
    let dataset = DummyDataset { data };
    let loader = DataLoader::new(&dataset, 3, false);

    let batches: Vec<Vec<i32>> = loader.iter().collect();

    assert_eq!(batches.len(), 4);
    assert_eq!(batches[0], vec![0, 1, 2]);
    assert_eq!(batches[1], vec![3, 4, 5]);
    assert_eq!(batches[2], vec![6, 7, 8]);
    assert_eq!(batches[3], vec![9]);
}

#[test]
fn test_dataloader_shuffle() {
    let data: Vec<i32> = (0..100).collect();
    let dataset = DummyDataset { data: data.clone() };
    let loader = DataLoader::new(&dataset, 10, true);

    let mut all_items: Vec<i32> = Vec::new();
    for batch in loader.iter() {
        all_items.extend(batch);
    }

    // Sort to verify all items are present
    let mut sorted_items = all_items.clone();
    sorted_items.sort();

    assert_eq!(sorted_items, data);

    // Likely not equal to original order (probabilistic but very likely)
    assert_ne!(all_items, data);
}
