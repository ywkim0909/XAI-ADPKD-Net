import numpy as np
from utils import *

import pickle

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.transform import resize

from lime import lime_image
from PIL import Image

mean_test = (0.05373301, 0.05373301, 0.05373301)
std_test = (0.11318668, 0.11318668, 0.11318668)

with open("model_weights/model_weights_adpkd_classification.pickle", "rb") as f_model:
    model_conv = pickle.load(f_model)

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


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=mean_test, std=std_test)

    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf


def batch_predict(images):
    model_conv.eval()
    preprocess_transform = get_preprocess_transform()

    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_conv.to(device)
    batch = batch.to(device)

    logits = model_conv(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def generate_xai_images():

    # put .tiff file path and name
    path = 'path/filename.tifff'

    img = generate_superpixels(path, 200, 10)

    pill_transf = get_pil_transform()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                             batch_predict, # classification function
                                             top_labels=3,
                                             hide_color=0,
                                             num_samples=10000,
                                             ) # number of images that will be sent to classification function

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=2,
                                                hide_rest=False,
                                               )
    resize_res = 224
    mask_disp = np.zeros((resize_res, resize_res, 3))
    mask_2d = resize(np.array(mask), (resize_res, resize_res), mode="edge", preserve_range=True)
    mask_disp[:, :, 1] = mask_2d

    mask_3d = np.zeros((224,224,3))
    mask_3d[:,:,1] = mask
    plt.imshow(temp)
    plt.imshow(mask_3d, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    generate_xai_images()
