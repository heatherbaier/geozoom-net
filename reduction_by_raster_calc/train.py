import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from torchvision.utils import make_grid, save_image
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import random
import torch
import json
import PIL


# Local imports
from handler import geozoom_handler
from fc_net import *
from utils import *
from sa import *


image_names = get_png_names("../data/MEX/")
y = get_migrants("../../pooling/data/migration_data.json" , image_names)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = models.resnet18().to(device)
attn_model = attnNet(resnet = resnet18).to(device)
lr = .0001
criterion = torch.nn.L1Loss(reduction = 'mean')
attn_optimizer = torch.optim.Adam(attn_model.parameters(), lr = lr)


butler = geozoom_handler(attn_model, 
                         device, 
                         criterion, 
                         attn_optimizer, 
                         num_thresholds = 4,
                         reduction_percent = .80,
                         convergence_dims = (224, 224),
                         change_bounds = (-50, 50),
                         perc_change_thresh = .65, 
                         plot = False, 
                         v = False)


BATCH_SIZE = 1
SPLIT = .60
train_dl, val_dl = butler.prep_attn_data(image_names, y, SPLIT, BATCH_SIZE)


butler.train_attn_model(train_dl, val_dl)