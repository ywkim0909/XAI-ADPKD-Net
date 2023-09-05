import numpy as np
import os
from skimage.segmentation import slic
from PIL import Image
from skimage.segmentation import mark_boundaries
import torchvision.transforms as transforms


def calculate_norm(dataset):
    # calculate means for axis=1, 2 of input dataset
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # calculate means for r, g, b channels
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # calculate std. dev. for axis 1, 2 of input dataset
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # calculate std. dev. for r, g, b channels
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

# load test image
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def generate_superpixels(image_path, n_segments, compactness):
    img = get_image(image_path)
    image_array = np.array(img)

    segments = slic(image_array, n_segments=n_segments, compactness=compactness)
    super_pix = mark_boundaries(image_array, segments)
    super_pix_uint8 = (super_pix * 255).astype(np.uint8)
    super_img = Image.fromarray(super_pix_uint8)
    return super_img


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
