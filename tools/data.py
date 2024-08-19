import os
import wget

# Define the URLs of the MNIST dataset files
urls = [
    "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
]

# Define the directory to save the downloaded files
directory = "/home/ubuntu/tannalwork/projects/ggml-mnist-project/data/"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Download the MNIST dataset files
for url in urls:
    filename = os.path.join(directory, os.path.basename(url))
    wget.download(url, filename)