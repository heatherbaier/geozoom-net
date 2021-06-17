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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# from utils import visualize_cam, Normalize, load_image, plot_gradcam, get_gradcam
# from gradcam import GradCAM, GradCAMpp

from attn_model2 import *

import matplotlib.pyplot as plt

from helpers import *
# from utils import *

from utils2 import *

from copy import deepcopy


class geozoom_handler():
    
    """
    Class to control the training of the geozoom-net
    Initialize at the bgeiining of training to handle all of the variables 
    """
    
    def __init__(self, model, device, criterion, optimizer, plot = False, v = False):
        
        self.image_sizes = {}
        self.distance_dict = {}
        self.diff_dict = {}
        self.epoch_train_loss = 0
        self.cur_threshold = 0
        self.epoch = 0
        self.loss_thresholds = []
        self.loss = 0
        self.threshold_index = 7
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.running_train_loss = 0
        self.running_val_loss = 0
        
        self.to_tens = transforms.ToTensor()
        
        self.plot = plot
        self.v = v
        
        self.best_weights = self.model.state_dict()
        self.best_loss = 90000000000000
        
        
        self.threshold_weights = {}
        
        self.convergence_dims = (224, 224)
        self.reduction_percent = .90
        
        
    def update_handler(self, train_dl):
        
        """
        General function that ingests the current status of the model and updates what needs to be updated (:
        """
                
        # After the first epoch, calculate all of the loss thresholds and set the beginning threshold index
        if self.epoch == 0:
            
            # Calculate 0, 20, 40, 60, & 80 percent loss thresholds
            self.loss_thresholds = self.__calc_loss_thresholds()
            self.threshold_index = 6
            self.cur_threshold = self.loss_thresholds[self.threshold_index]   
            
            print("Loss thresholds for training: ", self.loss_thresholds)
            print("Starting from threshold: ", self.threshold_index, " with value: ", self.cur_threshold)
            
        if self.epoch_train_loss < self.best_loss:    
            
#             print("  Updating model weights.")
            self.best_loss = self.epoch_train_loss
            self.best_weights = deepcopy(self.model.state_dict())
            
        # If the loss of the most recent epoch falls below the current threshold...
        if self.epoch_train_loss < self.cur_threshold:      
            
            self.threshold_weights[self.threshold_index] = self.best_weights
            self.best_weights = self.model.state_dict()
            self.best_loss = 90000000000000            
                        
            # Set the new threshold
            self.threshold_index -= 1
            self.cur_threshold = self.loss_thresholds[self.threshold_index]
            print("  Moving to threshold: ", self.threshold_index, "  |  Next loss benchmark: ", self.cur_threshold)
            
            # Update the sizes of the images in the dictionary (this happens each time we pass a threshold)
            self.update_image_sizes(train_dl)
         
        
    def __calc_loss_thresholds(self):
        
        """
        Function to calculate the loss thresholds for the trianing loop (these don't change after being calculated)
        """
        
        breaks = self.epoch_train_loss / 7
        breaks = [(i * breaks) for i in range(7)]
        return breaks
        
        
    def update_image_sizes(self, data):
        
        """
        Function to update the image sizes in the dictionary based on the gradcam heatmap for each of the images in the training dataset
        """
        
        # Iterate back over the images in the training dataset and clip them if needed
        for inp, out in data:

            # Grab the municipality ID from the image name and load it
            muni_id = inp[0].split("/")[4]
            
            # This will do the iterative clipping to the correct size
            cur_image = self.prep_input(inp)
            most_recent_im_size = (cur_image.shape[2], cur_image.shape[3])
            
            # Here, check to see if the image size is already less than or equal to the convergence dimensions,
            # If it is, skip the clipping and move on to the next input
            if (most_recent_im_size[0] <= self.convergence_dims[0]) or (most_recent_im_size[1] <= self.convergence_dims[1]):
                if self.v:
                    print("Image is already at scale, skipping clip.")
                continue                
            
            # If it's not the first epoch and the image needs to be clipped...
            if self.epoch > 0:
                    
                # Get the size of the image
                image_size = (cur_image.shape[2], cur_image.shape[3])
                            
                self.model.eval()
                
                # Get the gradcam and the attention heatmap for the current image
                gradcam, attn_heatmap = get_gradcam(self.model, image_size, cur_image.cuda())  
                
                # Then clip the input to the attention-based area
                cur_image, new_dims = self.clip_input(cur_image, attn_heatmap)
                
                if self.v:
                    print("\n")
                    print(muni_id)
                    print("old image size: ", image_size)
                    print("new image size: ", cur_image.shape[2], cur_image.shape[3])
                
                cur_image.cpu()
                
                # Update the image sizes in the dictionary
                self.image_sizes[muni_id].append(new_dims)
                                
                if self.plot:
                    plot_gradcam(gradcam)
                    plt.savefig(f"epoch{self.epoch}_muni{muni_id}_gradcam.png")

                
    def clip_input(self, image, attn_heatmap):
        
        """
        Function to clip the input based on the attention heatmap
        TO-DO: CLEAN UP THE VARIABLE NAMES HERE - THE LAST TIME YOU TRIED SOMETHING FUNKY HAPPENED (THE IMAGES DIDN'T ACTUALLY CLIP) SO COME BACK TO IT
        """
                
        # Get the indices of the max value in the heatmap
        max_index = np.where(attn_heatmap == np.max(attn_heatmap))
        
        # Calculate the dimensions that correspond to self.reduction_percent's of the most recent image size
        og_dims = (image.shape[2], image.shape[3])
        bounds = image.shape
        bounds = (int(bounds[2] * self.reduction_percent), int(bounds[3] * self.reduction_percent))
        
        # If the dimensions are less than the convergence dimensions, resize to convergence dims
        # TO-D0: MAKE A SELF.CONVERGENCE_DIMS PARAMETER
        if (bounds[0]) <= self.convergence_dims[0] or (bounds[1] <= self.convergence_dims[1]):
            bounds = (224, 224)
            
            
        rows, cols = bounds[0], bounds[1]
        ni, nj = attn_heatmap.shape 

        # If the number of indices that are greater than one, use the most central max index
        max_index = (max_index[0][int(len(max_index[0]) / 2)], max_index[1][int(len(max_index[1]) / 2)])
            
        rows_half = rows / 2
        cols_half = cols / 2
        
        """ ROWS """
        min_row = max(0, int(max_index[0] - rows_half))
        
        if min_row == 0:
            max_row = rows
            
        else:
            max_row = int(max_index[0] + rows_half)
            if max_row > og_dims[0]:
                min_row -= np.abs(max_row - og_dims[0])
                max_row = og_dims[0]
            
            
        """ COLUMNS """
        min_col = max(0, int(max_index[1] - cols_half))
        
        if min_col == 0:
            max_col = cols
            
        else:
            max_col = int(max_index[1] + cols_half)
            if max_col > og_dims[1]:
                min_col -= np.abs(max_col - og_dims[1])
                max_col = og_dims[1]
            
            
        indices = (min_row, max_row, min_col, max_col)
        image = image.detach().cpu()[:, :, min_row:max_row, min_col:max_col]

        return image, indices
        
        
    def prep_input(self, impath):
        
        """
        Function to load in an image and clip it to it's most updated size
        TO-DO: SEE IF YOU CAN DO THE CLIPPING WITHOUT DETACHING FROM THE GPU
        """
        
        muni_id = impath[0].split("/")[4]
        
        # Load the image as a tensor
        image = self.to_tens(Image.open(impath[0]).convert('RGB')).unsqueeze(0)
        
        # If the muni_id is in the image_sizes dictionary, clip it to the right size
        if muni_id in self.image_sizes.keys():
            for size in self.image_sizes[muni_id]:
                image = image.detach().cpu()[:, :, size[0]:size[1], size[2]:size[3]]
            
        if self.epoch == 0:
            self.image_sizes[muni_id] = [(0, image.shape[2], 0, image.shape[3])]
        
        return image.to(self.device)
        
        
        
    def get_original_size(self, impath):
        
        """
        Function to load in an image and clip it to it's most updated size
        TO-DO: SEE IF YOU CAN DO THE CLIPPING WITHOUT DETACHING FROM THE GPU
        """
        
        muni_id = impath[0].split("/")[4]
        
        # Load the image as a tensor
        image = self.to_tens(Image.open(impath[0]).convert('RGB')).unsqueeze(0)
        
        return image.to(self.device)        
        
        
    def calc_distances():
        pass
    
    
    def train(self, input, output):
        
        """
        Train model using an input
        """
        
        self.model.train()
        y_pred = self.model(input)    
        loss = self.criterion(y_pred, output.view(-1,1).to(self.device))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.running_train_loss += loss.item()
        

        

    def end_epoch(self, train_dl, val_dl):
        
        """
        Print a message to the user with the epoch stats and do all of the neccessary updates
        """
        
        self.epoch_train_loss = self.running_train_loss / len(train_dl)
        
        print("Epoch: ", self.epoch)
        print("  Training Loss: ", self.epoch_train_loss)
        
        
        self.update_handler(train_dl)
#         self.epoch_val_loss = self.running_val_loss / len(val_dl)

    
        self.running_train_loss = 0
#         self.running_val_loss = 0
        
        self.epoch += 1
        
        print("\n")
        
    def predict(self, input):
        
        """
        Function to predict a single datapoint
        """
        
        self.model.load_state_dict(self.best_weights)
        
        self.model.eval()
        y_pred = self.model(input)  
        
        return y_pred.item()
    
    def prep_attn_data(self, image_names, y, split, batch_size):
        
        x_train, y_train, x_val, y_val = train_test_split(np.array(image_names), np.array(y), split)

        train = [(k,v) for k,v in zip(x_train, y_train)]
        val = [(k,v) for k,v in zip(x_val, y_val)]

        train_dl = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
        val_dl = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)

        return train_dl, val_dl
    
    def train_attn_model(self, train_dl, val_dl):
        
        while self.threshold_index > 0:
            
            for impath, output in train_dl:

                # Prep the input and pass it to the trainer (this could easily be done in one step eventually if ya want)
                input = self.prep_input(impath)
                self.train(input, output)

            self.end_epoch(train_dl, val_dl = None)
            
            
#     def train_fc_model(self)

