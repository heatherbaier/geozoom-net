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


    

class fc_net(torch.nn.Module):
    
    def __init__(self, resnet):
        super(fc_net, self).__init__()

        # Normal resnet stuff
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = torch.nn.ReLU(inplace=False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.linear = torch.nn.Linear(in_features = 512, out_features = 1, bias = True)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.adp_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adp_pool(out)
        out = out.flatten(start_dim=1)
        out = self.linear(out)

        return out