pub trait Dataset {
    type Item;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Self::Item;
}
