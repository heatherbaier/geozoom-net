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

# from attn_model2 import *
# from helpers import *
from utils import *

# Estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor



class geozoom_handler():
    
    """
    Class to control the training of the geozoom-net
    Initialize at the beginning of training to handle all of the variables 
    """
    
    def __init__(self, 
                 model, 
                 device, 
                 criterion, 
                 optimizer, 
                 num_thresholds = 7,                 # Number of times (-1) to clip an image
                 convergence_dims = (224, 224),      # Smallest dimenions an image can be clipped to
                 reduction_percent = .70,            # Percentage of the image to keep each time it is clipped
                 change_bounds = (-100, 100),
                 perc_change_thresh = .60,
                 estimator = 'rf',
                 plot = False, 
                 v = False):
        
        
        self.device = device
        self.to_tens = transforms.ToTensor()
        self.plot = plot
        self.v = v      
        self.convergence_dims = convergence_dims
        self.reduction_percent = reduction_percent    
        
        self.num_thresholds = num_thresholds
        self.threshold_index = self.num_thresholds
        self.threshold_weights = {}
        self.loss_thresholds = []
        self.cur_threshold = 0
        self.image_sizes = {}
        
        self.epoch = 0
        self.loss = 0
        self.running_train_loss = 0
        self.running_val_loss = 0
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0             
        
        # Variables for the self-attention based spatial filtering model
        self.attn_model = model
        self.model = model
        self.criterion = criterion
        self.reset_optimizer = optimizer
        self.optimizer = self.reset_optimizer

        self.stage = 'attn'
        
        self.hm_tracker = {}
        self.num_pixels_same_dict = {}
        self.perc_change_dict = {}
        self.num_epoch_change_threshold = 2
        self.perc_change_thresh = perc_change_thresh
        self.change_bounds = change_bounds

        
        self.scale_estimates = {}
        self.estimator = estimator
        
        
    def update_handler(self, train_dl, val_dl):
        
        """
        General function that ingests the current state of the model and updates what needs to be updated... (:
        """
                
        # After the first epoch, calculate all of the loss thresholds and set the beginning threshold index
        if self.epoch == 0:
            
            # Calculate 0, 20, 40, 60, & 80 percent loss thresholds
            self.threshold_index = self.num_thresholds - 1
            
        # If the stats of the epoch pass the clip check:
        if self.clip_check():
            
            # Update the weights for the image reduction threshold
            self.threshold_weights[self.threshold_index] = deepcopy(self.model.state_dict())
                                    
            # Set the new threshold
            self.threshold_index -= 1
#             self.cur_threshold = self.loss_thresholds[self.threshold_index]
            print("  Moving to threshold: ", self.threshold_index)
            
            # Update the sizes of the images in the dictionary (this happens each time we pass a threshold)
            self.update_image_sizes(train_dl)
            self.update_image_sizes(val_dl)
            self.optimizer = self.reset_optimizer
            
            self.hm_tracker = {}
            self.num_pixels_same_dict = {}
            self.perc_change_dict = {}
            
            self.save_scale_estimates(train_dl, val_dl)
                        
#             print("Reinitializing random weights.")
#             self.model = self.attn_model


    def save_scale_estimates(self, train_dl, val_dl):
        
        self.model.eval()
        
        for impath, output in train_dl:   
            muni_id = impath[0].split("/")[3]
            input = self.prep_input(impath)
            self.stats(impath, input)
            y_pred = self.model(input).item()
            
            if muni_id not in self.scale_estimates.keys():
                self.scale_estimates[muni_id] = [y_pred]
            else:
                self.scale_estimates[muni_id].append(y_pred)

        for impath, output in val_dl:
            muni_id = impath[0].split("/")[3]
            input = self.prep_input(impath)
            self.val(input, output)
            y_pred = self.model(input).item()
            
            if muni_id not in self.scale_estimates.keys():
                self.scale_estimates[muni_id] = [y_pred]
            else:
                self.scale_estimates[muni_id].append(y_pred)
                

    def clip_check(self):
        
        """
        Function to check if the attention maps pass the input parameters and the images are ready to clip
        """
        
        if len(self.perc_change_dict) >= 1:
        
            vals = list(self.perc_change_dict.values())

            if len(vals[0]) > self.num_epoch_change_threshold:

                if self.num_epoch_change_threshold >= len(vals[0]):
                    vals = [i for i in vals]

                else:
                    vals = [i[(-self.num_epoch_change_threshold - 1):-1] for i in vals]
                
                vals = np.array([np.max(i) for i in vals])

                
                
                num_change_thresh = .6
                
                
                

                num_above = vals[(vals >= self.change_bounds[0]) & (vals <= self.change_bounds[1])].shape[0]
                perc_above = (num_above / len(vals))

                print("Percentage above threshold: ", perc_above)

                if perc_above > self.perc_change_thresh:
                    return True
                else:
                    return False
                
            else:
            
                return False
            
    
        else:
            
            return False
            
                    
        
            
        
    def __calc_loss_thresholds(self):
        
        """
        Function to calculate the loss thresholds for the trianing loop (these don't change after being calculated)
        """
        
        breaks = self.epoch_val_loss / self.num_thresholds
        breaks = [(i * breaks) for i in range(self.num_thresholds)]
        return breaks
        
                
    def update_image_sizes(self, data):
        
        """
        Function to update the image sizes in the dictionary based on the gradcam heatmap for each of the images in the training dataset
        """
        
        # Iterate back over the images in the training dataset and clip them if needed
        for inp, out in data:

            # Grab the municipality ID from the image name and load it
            muni_id = inp[0].split("/")[3]
            
            # This will do the iterative clipping to the correct size
            cur_image = self.prep_input(inp)
            most_recent_im_size = (cur_image.shape[2], cur_image.shape[3])
            
            # Here, check to see if the image size is already less than or equal to the convergence dimensions,
            # If it is, skip the clipping and move on to the next input
            if (most_recent_im_size[0] <= self.convergence_dims[0]) or (most_recent_im_size[1] <= self.convergence_dims[1]):
                if self.v:
                    print("Image is already at scale, skipping clip.")
                continue                
            
            # If it's not the first epoch...
            if self.epoch > 0:
                
                self.model.eval()
                    
                # Get the size of the image
                image_size = (cur_image.shape[2], cur_image.shape[3])
                                            
                # Get the gradcam and the attention heatmap for the current image
                gradcam, attn_heatmap = get_gradcam(self.model, image_size, cur_image.cuda(), target_layer = self.model.sa)  
                
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
    
    
    
    def stats(self, impath, cur_image):
        
        muni_id = impath[0].split("/")[3]
        
        # Get the size of the image
        image_size = (cur_image.shape[2], cur_image.shape[3])

        # Get the gradcam and the attention heatmap for the current image
        gradcam, attn_heatmap = get_gradcam(self.model, image_size, cur_image.cuda(), target_layer = self.model.sa)  
        
        breaks = [i * (255 / 4) for i in range(0, 5)]
        attn_heatmap[(attn_heatmap >= breaks[0]) & (attn_heatmap < breaks[3])] = 1
        attn_heatmap[(attn_heatmap >= breaks[3]) & (attn_heatmap < breaks[4])] = 255
        
        
        if self.plot:
            plt.imshow(attn_heatmap)
            plt.savefig(f"{muni_id}_epoch{self.epoch}.png")
            plt.clf()
        
        
        # If there is no previous heatmap already saved for a given municipality, save the heatmap and move on 
        # i.e. if self.epoch == 0...
        if muni_id not in self.hm_tracker:
            self.hm_tracker[muni_id] = [attn_heatmap]
            return 
        
        # If a previous heatmap does exist...
        else:
            
            # Grab the most recent heatmap
            hm_t1 = self.hm_tracker[muni_id][-1]
            hm_t2 = attn_heatmap
            
            # Calculate the number of pixels that are the same in the highest attention 
            # class between the two images
            num_pixels_same_class1 = hm_t1[(hm_t1 == 255) & (hm_t2 == 255)].shape
            
            # If the image is not already in the num_pixels_same_dict, save it and return
            if muni_id not in self.num_pixels_same_dict:
                self.num_pixels_same_dict[muni_id] = [num_pixels_same_class1[0]]
                return
            
            # If it does exist in the dictionary...
            else:
                
                # Append the number of same pixels
                self.num_pixels_same_dict[muni_id].append(num_pixels_same_class1[0])
                
                # If the number of pixels saved is more than 1...
                if len(self.num_pixels_same_dict[muni_id]) > 1:
                    
                    # Calculate the percentage change
                    nps_t1 = self.num_pixels_same_dict[muni_id][-2]
                    nps_t2 = self.num_pixels_same_dict[muni_id][-1]
                    
                    # Potential divsion by zero error
                    try:
                        perc_change = ((nps_t2 - nps_t1) / nps_t2) * 100
                    except:
                        perc_change = 0
                    
                    if muni_id not in self.perc_change_dict:
                        self.perc_change_dict[muni_id] = [perc_change]
                    
                    else:
                        self.perc_change_dict[muni_id].append(perc_change)
                    
                
    def prep_input(self, impath):
        
        """
        Function to load in an image and clip it to it's most updated size
        TO-DO: SEE IF YOU CAN DO THE CLIPPING WITHOUT DETACHING FROM THE GPU
        """
        
        muni_id = impath[0].split("/")[3]
        
        # Load the image as a tensor
        image = self.to_tens(Image.open(impath[0]).convert('RGB')).unsqueeze(0)
        
        # If the muni_id is in the image_sizes dictionary, clip it to the right size
        if muni_id in self.image_sizes.keys():
            for size in self.image_sizes[muni_id]:
                image = image.detach().cpu()[:, :, size[0]:size[1], size[2]:size[3]]
            
        if self.stage == 'attn':
            if self.epoch == 0:
                self.image_sizes[muni_id] = [(0, image.shape[2], 0, image.shape[3])]
        
        return image.to(self.device)
    
    
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
        
        
    def val(self, cur_image, output):
        
        """
        Pass a validation image through the trained thresholds
        """
        
        self.model.eval()
        y_pred = self.model(cur_image.cuda())
        loss = self.criterion(y_pred, output.view(-1,1).to(self.device))
        self.running_val_loss += loss.item()


    def end_epoch(self, train_dl, val_dl):
        
        """
        Print a message to the user with the epoch stats and do all of the neccessary updates
        """
        
        self.epoch_train_loss = self.running_train_loss / len(train_dl)
        self.epoch_val_loss = self.running_val_loss / len(val_dl)
        
        print("Epoch: ", self.epoch)
        print("  Training Loss: ", self.epoch_train_loss)
        print("  Validation Loss: ", self.epoch_val_loss)
        
        if self.stage == 'attn':
            self.update_handler(train_dl, val_dl)
            
            
        if self.stage == 'fc':
            if self.epoch_val_loss < self.best_loss:
                print("Updating best weights!")
                self.best_loss = self.epoch_val_loss
                self.best_weights = deepcopy(self.model.state_dict())

        self.running_train_loss = 0
        self.running_val_loss = 0
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
        
        """
        Function to split data into training and validation dataloaders
        Called from outside of the class beofre model is training
        """
        
        x_train, y_train, x_val, y_val = train_test_split(np.array(image_names), np.array(y), split)

        train = [(k,v) for k,v in zip(x_train, y_train)]
        val = [(k,v) for k,v in zip(x_val, y_val)]

        train_dl = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
        val_dl = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)

        return train_dl, val_dl

    
    def train_attn_model(self, train_dl, val_dl):
        
        """
        Function to train the self-attention based spatial filtering part of the model
        When the trehshold index hits 0, the function passes to the fully connected model training
        """
        
        # While the threshold index is above 0, train the self-attention based spatial filtering mode
        # to get the standard iamge sizes for the fc model
        while self.threshold_index > 0:
            
            for impath, output in train_dl:                
                input = self.prep_input(impath)
                self.stats(impath, input)
                self.train(input, output)
                
            for impath, output in val_dl:
                input = self.prep_input(impath)
                self.val(input, output)

            self.end_epoch(train_dl, val_dl)
            
            
        # When the trehshold index hits 0, move on to training the fc model with no attention layer
        if self.threshold_index == 0:
            
#             print("\nImage sizes entering FC model: \n")
#             print(self.image_sizes)
            print("Switching to fully connected model!")
            self.train_fc_model(train_dl, val_dl)
            
    
    def train_fc_model(self, train_dl, val_dl, outside_estimator = None):
        
        """
        Function to train the fully connected model on the standard image sizes after the self-attention based spatial filtering model has finishing filtering down each of the images
        """
        
        # Orgnanize data for tabular model
        y_dict = {}
        
        for impath, output in train_dl:
            muni_id = impath[0].split("/")[3]
            y_dict[muni_id] = output.numpy()[0]
        for impath, output in val_dl:
            muni_id = impath[0].split("/")[3]
            y_dict[muni_id] = output.numpy()[0]

        muni_ids = list(y_dict.keys())
        
        
        x = [self.scale_estimates[mid] for mid in muni_ids]
        y = [y_dict[mid] for mid in muni_ids]
        
        if  outside_estimator == None:
            estimator = self.estimator
        else:
            estimator = outside_estimator
            
            
        if estimator == 'rf':
        
            rfr = RandomForestRegressor()
            rfr.fit(x, y)
            print("Random Forest MAE: ", mae(y, rfr.predict(x)))
            
        elif estimator == 'dt':
        
            dtr = DecisionTreeRegressor(max_depth = 5)
            dtr.fit(x, y)
            print("Decision Tree MAE: ", mae(y, dtr.predict(x)))    
            
        elif estimator == 'mlp':
        
            mlp = MLPRegressor()
            mlp.fit(x, y)
            print("MLP MAE: ", mae(y, mlp.predict(x)))  
            
        elif estimator == 'knn':
        
            knn = KNeighborsRegressor()
            knn.fit(x, y)
            print("KNN MAE: ", mae(y, knn.predict(x)))              
            

    def plot_attn_map(self, impath):
        
        """
        Function to save the attention maps for a given image at each threshold
        """
        
        muni_id = impath.split("/")[3]
        cur_image = self.prep_input([impath])
        folder = f"{muni_id}_attention_maps"
        
        if not os.path.isdir(folder):
            os.mkdir(folder)
            

        for k in self.threshold_weights.keys():

            if k != 'fc':

                model = self.attn_model
                model.load_state_dict(self.threshold_weights[k])
                model.eval()
                IM_SIZE = (cur_image.shape[2], cur_image.shape[3])
                gradcam, attn_heatmap = get_gradcam(model, IM_SIZE, cur_image.cuda(), target_layer = self.model.sa) 
                cur_image, new_dims = self.clip_input(cur_image, attn_heatmap)

                fname = os.path.join(f"{muni_id}_attention_maps", f"threshold{k}_muni{muni_id}.png") 
                
                plot_gradcam(gradcam)
                plt.savefig(fname)
                plt.clf()

