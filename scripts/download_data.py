import os
import urllib.request
import gzip
import shutil

# datasets config
DATASETS = {
    "mnist": {
        "base_url": "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "dir": "data/mnist",
        "files": [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]
    },
    "fashion-mnist": {
        "base_url": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "dir": "data/fashion-mnist",
        "files": [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]
    }
}

def download_and_extract(dataset_name, config):
    data_dir = config["dir"]
    base_url = config["base_url"]
    files = config["files"]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"[{dataset_name}] Created directory: {data_dir}")

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        url = base_url + filename
        
        # Download
        if not os.path.exists(file_path):
            print(f"[{dataset_name}] Downloading {filename} from {url}...")
            try:
                # Add user-agent to avoid 403 in some cases
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                print(f"[{dataset_name}] Downloaded {filename}")
            except Exception as e:
                print(f"[{dataset_name}] Failed to download {filename}: {e}")
                continue
        else:
            print(f"[{dataset_name}] {filename} already exists.")

        # Extract (optional, but requested by user pattern)
        # Keeping .gz files is enough for fulaminas generic loader, 
        # but extracting might be useful for other tools.
        # output_filename = filename[:-3]
        # output_path = os.path.join(data_dir, output_filename)
        
        # if not os.path.exists(output_path):
        #     print(f"[{dataset_name}] Extracting {filename}...")
        #     with gzip.open(file_path, 'rb') as f_in:
        #         with open(output_path, 'wb') as f_out:
        #             shutil.copyfileobj(f_in, f_out)
        #     print(f"[{dataset_name}] Extracted to {output_filename}")

if __name__ == "__main__":
    for name, config in DATASETS.items():
        download_and_extract(name, config)
