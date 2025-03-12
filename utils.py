import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

from sklearn.datasets import fetch_openml

def get_images(dataset_type, class_val):
    """
    Downloads images and flattens them into vectors. For RGB images (CIFAR_10),
    converts into grayscale by averaging the RGB channels.

    :param dataset_type: a string representing the name of the dataset for which
    images should be downloaded. Should be either "CIFAR_10", "mnist_784", or "Fashion-MNIST".
    :param class_val: an integer representing the class for which images should be downloaded.
    :return: a Numpy array representing the flatteend image dataset
    of shape (n_images, flattened_img_dim) where flattened_img_dim is the "length * width" of each image.
    """
    # dataset_type = "CIFAR_10" | "mnist_784" | "Fashion-MNIST"
    if (dataset_type == "CIFAR_10"):
        # Define a transform that converts the image to grayscale by averaging the RGB channels
        class RGB2GrayTransform:
            def __call__(self, img):
                img = torch.mean(img, dim=0, keepdim=True)  # Average the color channels
                return img

        # Load CIFAR-10 dataset with the grayscale transform
        transform = transforms.Compose([
            transforms.ToTensor()         # Convert to Tensor
        ])
        
        # Download CIFAR data into the training directory
        cifar10_train = datasets.CIFAR10(root='../training_data', train=True, download=True, transform=transform)

        class_0_images = [img for img, label in cifar10_train if label == class_val]
        
        # Convert each image to grayscale, flatten it, and store it in a list
        grayscale_flattened_images = []

        for img in class_0_images:
            # Convert to grayscale by averaging the RGB channels (shape becomes (32, 32))
            img_gray = torch.mean(img, dim=0)
            
            # Flatten the grayscale image (32*32 = 1024 pixels)
            img_gray_flat = img_gray.view(-1).numpy()
            
            # Append the flattened image to the list
            grayscale_flattened_images.append(img_gray_flat)

        # Convert the list to a NumPy array with shape (n_images, dimension)
        class_0_array = np.array(grayscale_flattened_images)

        print(class_0_array.shape)  # Should output (n_images, 1024)
        return class_0_array
    else:
        # Load the MNIST dataset
        mnist = fetch_openml(dataset_type, version=1)
        X = mnist.data  # Shape: (70000, 784)
        y = mnist.target.astype(int)
        # Filter out the digit '0'
        X_zero = X[y == class_val]
        return X_zero.to_numpy() / 255.0