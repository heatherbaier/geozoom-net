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

import json

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from utils import visualize_cam, Normalize, load_image, plot_gradcam, get_gradcam
# from gradcam import GradCAM, GradCAMpp

from attn_model import *

import matplotlib.pyplot as plt

from helpers import *
# from utils import *

from utils import *


image_names = get_png_names("../pooling/data/MEX2/")#[0:5]
y = get_migrants("../pooling/data/migration_data.json" , image_names)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = models.resnet18().to(device)
model = attnNetBinary(in_channels = 512, h = 7, w = 7, batch_size = 1, resnet = resnet18).to(device)
lr = .0001
criterion = torch.nn.L1Loss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

BATCH_SIZE = 1

x_train, y_train, x_val, y_val = train_test_split(np.array(image_names), np.array(y), .70)

train = [(k,v) for k,v in zip(x_train, y_train)]
val = [(k,v) for k,v in zip(x_val, y_val)]

train_dl = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)


print("Training: ", len(train_dl))
print("Validation: ", len(val_dl))

model.train()

for i in range(200):
        
    running_train_loss, running_val_loss = 0, 0
    
    for input, output in train_dl:
                
        input = load_inputs(input[0]).to(device)
        model.train()
        y_pred = model(input)    
        loss = criterion(y_pred, output.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        
    for input, output in val_dl:
                
        input = load_inputs(input[0]).to(device)
        y_pred = model(input)    
        loss = criterion(y_pred, output.to(device))
        
        running_val_loss += loss.item()

    print("Epoch: ", i)
    print("  Train Loss: ", running_train_loss / len(train_dl))
    print("  Val Loss: ", running_val_loss / len(val_dl))
    print("\n")