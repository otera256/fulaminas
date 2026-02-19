# Setup Guide

## Data Preparation

To run examples involving MNIST or Fashion-MNIST, you need to download the datasets first.
We provide a Python script to do this reliably using stable mirrors.

### Prerequisites

- Python 3.x installed.

### Steps

1. Run the download script:
   ```bash
   python scripts/download_data.py
   ```
   This will create a `data` directory and download:
   - `data/mnist`: MNIST dataset
   - `data/fashion-mnist`: Fashion-MNIST dataset

2. Now you can run the Rust examples:
   ```bash
   cargo run --example view_mnist
   ```
