import numpy as np
from pathlib import Path
from utils import *

import pickle

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

def load_weights_and_classify():
    img_folder = 'test_img/'
    with open("model_weights/model_weights_adpkd_classification.pickle", "rb") as f_model:
        model_conv = pickle.load(f_model)

    resize_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    resize_test = torchvision.datasets.ImageFolder(root=img_folder, transform=resize_transforms)
    mean_test, std_test = calculate_norm(resize_test)

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean_test, std_test)
    ])

    classes = ('Class_2', 'Class_1')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_conv.to(device)
    print(device)

    # name of images
    img_list = Path('test_img/class_2').glob("*.tiff")
    img_names = []
    for name in img_list:
        name_tmp_img = Path(name).stem
        img_names.append(name_tmp_img)
    img_names = np.array(sorted(img_names), dtype="str")

    # path of images
    load_path_img = []
    for i in img_names:
        load_tmp_path = img_folder+"class_2/" + i + ".tiff"
        load_path_img.append(load_tmp_path)
    load_path_img = np.array(sorted(load_path_img), dtype="str")

    # "2A" classification 확률
    for i in range(len(load_path_img)):
        image = pil_loader((load_path_img[i]))
        image = transform_test(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model_conv(image)
            _, preds = torch.max(outputs, 1)
            softmax_fn = F.softmax(outputs, 1)
            print('Case name {} is {} with classification confidence of  {:.1f}%'.format(img_names[i], classes[preds[0]], softmax_fn[0][preds[0]]*100))


if __name__ == '__main__':
    load_weights_and_classify()