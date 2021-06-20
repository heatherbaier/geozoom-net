from torchvision.utils import make_grid, save_image
from torchvision import transforms, utils, models
from gradcam import GradCAM, GradCAMpp
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
from torch import nn
import pandas as pd
import numpy as np
import torchvision
import random
import torch
import math
import json
import time
import cv2
import PIL
import os


def visualize_cam(mask, img):
    
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    
    mask = mask.cpu()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result, np.uint8(255 * mask.squeeze())


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_image(img_path):
    
    """
    Function to load a single image as a 4D tensor
    """
    
    pil_img = PIL.Image.open(img_path)
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
    IM_SIZE = (torch_img.shape[2], torch_img.shape[3])
    return torch_img, IM_SIZE


def plot_gradcam(images, im_name = None):
    
    """
    Function to plot the original image, gradcam heatmap and heatmap overtop of the original image
    """
    
    plt.figure(figsize = (30, 30))
    a = 0
    titles = ['Image', 'Heatmap', 'Result']
    for i in images[0]:
        plt.subplot(1, 3, a + 1)
        plt.imshow(i.permute(1, 2, 0))
        plt.title(titles[a])
        plt.axis("off")
        a += 1
    if im_name != None:
        plt.savefig(im_name)


def get_gradcam(model, im_size, image, target_layer):
    
    """
    Function to get the gradcam & heatmap for an inpiut image
    """

    cam_dict = dict()
    resnet_model_dict = dict(type='resnet', arch = model, layer_name = 'layer4', input_size = (im_size[0], im_size[1]), target_layer = target_layer)
    resnet_gradcam = GradCAM(resnet_model_dict, True)
    resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

    images = []
    for gradcam, gradcam_pp in cam_dict.values():
        # print(gradcam)
        mask, _ = gradcam(image, im_size = im_size)
        heatmap, result, mask = visualize_cam(mask.cpu(), image.cpu())
        images.append(torch.stack([image.squeeze().cpu(), heatmap, result], 0))

    return images, mask



def load_inputs(impath):
    
    """
    Function to load a single image as a 4D tensor
    """
    
    to_tens = transforms.ToTensor()
    return to_tens(Image.open(impath).convert('RGB')).unsqueeze(0)


def get_png_names(directory):
    
    """
    Get the paths to the files available file input into the model
    """
    
    images = []
    for i in os.listdir(directory):
        try:
            if os.path.isdir(os.path.join(directory, i)):
                new_path = os.path.join(directory, i, "pngs")
                image = os.listdir(new_path)[0]
                images.append(os.path.join(directory, i, "pngs", image))
        except:
            pass
    return images
    
    
def get_migrants(json_path, images):
    
    """
    Get a list of the y features corresponding to each image in image_names
    """
    
    m = open(json_path,)
    mig_data = json.load(m)
    m.close()
    muni_ids = [i.split("/")[5] for i in images]
    print(len(muni_ids), "municipalities.")  
    return [mig_data[i] for i in muni_ids]


def train_test_split(image_names, y, split):
    
    """
    Train, test, split the data
    """

    train_num = int(len(image_names) * split)
    val_num = int(len(image_names) - train_num)

    all_indices = list(range(0, len(image_names)))
    train_indices = random.sample(range(len(image_names)), train_num)
    val_indices = list(np.setdiff1d(all_indices, train_indices))

    x_train, x_val = image_names[train_indices], image_names[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return x_train, y_train, x_val, y_val

