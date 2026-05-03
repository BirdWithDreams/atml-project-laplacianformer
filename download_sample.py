import os
import urllib.request
import tarfile
import shutil

def download_and_extract_imagenette(data_dir="./data/imagenet"):
    # Imagenette is a subset of 10 easily classified classes from ImageNet
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    tar_path = "./data/imagenette.tgz"
    
    os.makedirs("./data", exist_ok=True)
    
    print(f"Downloading Imagenette sample from {url}...")
    urllib.request.urlretrieve(url, tar_path)
    
    print("Extracting dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path="./data")
        
    # Rename the extracted folder to match your datamodule's expected path
    extracted_folder = "./data/imagenette2-320"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.rename(extracted_folder, data_dir)
    
    # Clean up the tar file
    os.remove(tar_path)
    print(f"Success! ImageNet sample is ready at: {data_dir}")

if __name__ == "__main__":
    download_and_extract_imagenette()