import numpy as np
import random
import torch

import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

import os
import json


from torchvision import transforms, utils, models
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
import os


def load_inputs(impath):
    to_tens = transforms.ToTensor()
    return to_tens(Image.open(impath).convert('RGB')).unsqueeze(0)


def get_png_names(directory):
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
    m = open(json_path,)
    mig_data = json.load(m)
    m.close()
    muni_ids = [i.split("/")[4] for i in images]
    print(len(muni_ids), "municipalities.")  
    return [mig_data[i] for i in muni_ids]


def train_test_split(image_names, y, split):

    train_num = int(len(image_names) * split)
    val_num = int(len(image_names) - train_num)

    all_indices = list(range(0, len(image_names)))
    train_indices = random.sample(range(len(image_names)), train_num)
    val_indices = list(np.setdiff1d(all_indices, train_indices))

    x_train, x_val = image_names[train_indices], image_names[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return x_train, y_train, x_val, y_val


def prep_batch(batch):
    
    to_tens = transforms.ToTensor()
    batch = [to_tens(Image.open(i).convert('RGB')).unsqueeze(0) for i in batch]
    batch_w = [i.shape[2] for i in batch]
    batch_h = [i.shape[3] for i in batch]
    min_w = min(batch_w)
    min_h = min(batch_h)
    batch_max = min(min_w, min_h)
    batch_min = max(0, batch_max - 100)
    batch_size0 = random.randint(batch_min, batch_max)
    batch_size1 = random.randint(batch_min, batch_max)                                 
    rc = transforms.RandomCrop((batch_size0, batch_size1))
    
    batch_out = [rc(i) for i in batch]
    batch_out = torch.tensor(torch.cat(batch_out))
    min_mean = torch.min(torch.mean(batch_out, dim = (1,2,3)))
    
    while_num = 0
    while (min_mean < .001) & (while_num < 50):
        batch_out = [rc(i) for i in batch]
        batch_out = torch.tensor(torch.cat(batch_out))
        min_mean = torch.min(torch.mean(batch_out, dim = (1,2,3)))  
        while_num += 1
            
    return batch_out