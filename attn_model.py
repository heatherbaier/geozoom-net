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

# from utils import visualize_cam, Normalize
# from gradcam import GradCAM, GradCAMpp



class channelMaxPool(torch.nn.Module):
    def __init__(self, in_channels, h, w, batch_size):
        super(channelMaxPool, self).__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = h


    def forward(self, x):
        # torch.reshape(v, (1,2,2))
        x, i = torch.max(x, dim = 1)
        return torch.reshape(x, (x.shape[0], 1, self.h, self.w))


class spatialMaxPool(torch.nn.Module):
    def __init__(self, in_channels, batch_size):
        super(spatialMaxPool, self).__init__()
        self.in_channels = in_channels
        self.batch_size = batch_size

    def forward(self, x):
        x, i = torch.max(x, dim = -1)
        x, i = torch.max(x, dim = -1)
        return torch.reshape(x, (x.shape[0], self.in_channels, 1, 1))


class channelAvgPool(torch.nn.Module):
    def __init__(self, in_channels, h, w, batch_size):
        super(channelAvgPool, self).__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = h


    def forward(self, x):
        x = torch.mean(x, dim = 1)
        return torch.reshape(x, (x.shape[0], 1, self.h, self.w))


class spatialAvgPool(torch.nn.Module):
    def __init__(self, in_channels, batch_size):
        super(spatialAvgPool, self).__init__()
        self.in_channels = in_channels
        self.batch_size = batch_size

    def forward(self, x):
        x = torch.mean(x, dim = -1)
        x = torch.mean(x, dim = -1)
        return torch.reshape(x, (x.shape[0], self.in_channels, 1, 1))



class attnNetBinary(torch.nn.Module):
    def __init__(self, in_channels, h, w, batch_size, resnet):
        super(attnNetBinary, self).__init__()

        # Normal resnet stuff
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = torch.nn.ReLU(inplace=False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.linear = torch.nn.Linear(in_features=512, out_features = 1, bias = True)


        # Attention layers
        self.sMP = spatialMaxPool(in_channels = 512, batch_size = batch_size)
        self.cMP = channelMaxPool(in_channels = 512, h = 7, w = 7, batch_size = batch_size)
        self.sAP = spatialAvgPool(in_channels = 512, batch_size = batch_size)
        self.cAP = channelAvgPool(in_channels = 512, h = 7, w = 7, batch_size = batch_size)
        # self.out_channels = int(in_channels/16)
        self.out_channels = in_channels
        self.convR_M = torch.nn.Conv2d(in_channels = 512, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convA_M = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convB_M = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (3,3), bias=True, padding = 1)
        self.convC_M = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (7,7), bias=True, padding = 3)
        self.convE_M = torch.nn.Conv2d(in_channels = self.out_channels * 3, out_channels = 512, kernel_size = (1,1), bias=True)
        
        self.convR_A = torch.nn.Conv2d(in_channels = 512, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convA_A = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convB_A = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (3,3), bias=True, padding = 1)
        self.convC_A = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (7,7), bias=True, padding = 3)
        self.convE_A = torch.nn.Conv2d(in_channels = self.out_channels * 3, out_channels = 512, kernel_size = (1,1), bias=True)

        self.bn2 = torch.nn.BatchNorm2d(512)
        self.sigmoid = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax()
        self.adp_pool = torch.nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # save = deepcopy(out.detach())
        out = self.relu(out)
        out = self.bn2(out)

        out = self.adp_pool(out)

        # Max Pooling
        fsM = self.sMP(out)
        fcM = self.cMP(out)
        fscM = torch.mul(fsM, fcM)
        rM = self.convR_M(fscM)
        aM = self.convA_M(rM)
        bM = self.convB_M(rM)
        cM = self.convC_M(rM)
        catM = torch.cat((aM,bM,cM), dim = 1)
        eM = self.convE_M(catM)


        # Avg Pooling
        fsA = self.sAP(out)
        fcA = self.cAP(out)
        fscA = torch.mul(fsA, fcA)
        rA = self.convR_A(fscA)
        aA = self.convA_A(rA)
        bA = self.convB_A(rA)
        cA = self.convC_A(rA)
        catA = torch.cat((aA,bA,cA), dim = 1)
        eA = self.convE_A(catA)

        added = self.relu(torch.add(eA, eM))

        # attn_mask = self.relu(added)

        # out = torch.add(out, attn_mask)

        out = torch.add(out, added)

        out = self.avgpool(out)

        out = out.flatten(start_dim=1)
        out = self.linear(out)

#         out = self.sm(out)

        return out#, added#, save