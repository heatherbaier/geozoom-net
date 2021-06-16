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
    Class to control the training of the ZoomNet
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
            
            print("  Updating model weights.")
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
        
        # For each of the images in the training dataset
        # inp = (img_name,)
        # out = y
        for inp, out in data:

            # Grab the municipality ID from the image name and load the image (TO-DO: PUT LOAD_INPUTS IN THIS CLASS)
            muni_id = inp[0].split("/")[4]
            cur_image = self.prep_input(inp)
                        
            # If it's the first epoch, add the muni_id to the dictionary and the size of the original image in format: (x_min, x_max, y_min, y_max)
            if self.epoch == 0:
                if self.v:
                    print("In update, epoch is zero so adding image sizes.")
                self.image_sizes[muni_id] = (0, cur_image.shape[0], 0, cur_image.shape[1])
                continue
                                

            # If the municipality is already in the image_sizes dict, that means it has already been see and the image_size might be different
            # from the original, so grab the most updated image size from the dictionary and clip the image to those dimensions
            if muni_id in self.image_sizes.keys():
                                
                cur_im = self.prep_input(inp).to(self.device) 
                prev_im_size = self.image_sizes[muni_id]
                
                if (prev_im_size[1] == 224) or (prev_im_size[2] == 224):
                    
                    if self.v:
                        print("Image is already at scale, skipping clip.")
                    
                    continue                           
                    
                    
            # If it's past the first epoch...
            if self.epoch != 0:
#             else:
                    
                # Get the size of the image
                IM_SIZE = (cur_image.shape[2], cur_image.shape[3])
                
                self.model.eval()
                
                # Get the gradcam and the attention heatmap for the current image
                gradcam, attn_heatmap = get_gradcam(self.model, IM_SIZE, cur_image.cuda())  
                cur_image, new_dims = self.clip_input(cur_image, attn_heatmap)
                
                if self.v:
                    print("\n")
                    print(muni_id)
                    print("old image size: ", IM_SIZE)
                    print("new image size: ", cur_image.shape[2], cur_image.shape[3])
                
                cur_image.cpu()
                
                # Update the image sizes in the dictionary
                self.image_sizes[muni_id] = new_dims  
                
                if self.plot:
                    plot_gradcam(gradcam)
                    plt.savefig(f"epoch{self.epoch}_muni{muni_id}_gradcam.png")

                
    def clip_input(self, input, test_arr):
        
        """
        Function to clip the input based on the attention heatmap
        TO-DO: CLEAN UP THE VARIABLE NAMES HERE - THE LAST TIME YOU TRIED SOMETHING FUNKY HAPPENED (THE IMAGES DIDIN'T ACTUALLY CLIP) SO COME BACK TO IT
        """
                
        # Get the indices of the heatmap of the max value of the heatmap
        result = np.where(test_arr == np.max(test_arr))
        
        # Calculate the dimensions that correspond to X% of the most recent image size
        sh = test_arr.shape
        sh = (int(sh[0] * .70), int(sh[1] * .70))
        
        if (sh[0]) <= 224 or (sh[1] <= 224):
            if self.v:
                print("Image is already at scale, skipping clip.")
            return input.detach().cpu()[:, :, 0:224, 0:224], (0, 224, 0, 224)

        left, right = sh[0], sh[1]
        ni, nj = test_arr.shape 

        min_row, max_row = 50000, 0
        min_col, max_col = 50000, 0

        # If the number of indices that are the max are exxcessive, just use the first
        # TO-DO: CONSIER A BETTER WAY OF DOING THIS
        if (len(result[0]) > 100) or (len(result[1])  > 100):
            result = (np.array([result[0][0]]), np.array([result[1][0]]))
            
        # For every x,y max index, do stuff (:
        for i, j in zip(result[0], result[1]):

            istart, istop = max(0, i-left), min(left, i+left+1)
            jstart, jstop = max(0, j-right), min(right, j+right+1)            

            if istart < min_row:
                min_row = istart
            if jstart < min_col:
                min_col = jstart   
            if istop > max_row:
                max_row = istop    
            if jstop > max_col:
                max_col = jstop  
                
        indices = (min_row, max_row, min_col, max_col)
        if (indices[1]) < 224 or (indices[3] < 224):
            """ TO-DO: I THINK THIS NEEDS TO BE MORE THOROUGHLY THOUGHT OUT LATER ON """
            if self.v:
                print("Image size is now below 224. Resizing back up to 224x224")
            indices = (min_row, 224, min_col, 224)
        
        input = input.detach().cpu()[:, :, min_row:max_row, min_col:max_col]

        return input, indices
        
        
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
        
            dims = self.image_sizes[muni_id]
            image = image.detach().cpu()[:, :, dims[0]:dims[1], dims[2]:dims[3]]
        
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