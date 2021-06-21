import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

from utils import *



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
                 fc_model,
                 fc_optimizer,
                 num_fc_epochs = 100,
                 fc_batch_size = 2,
                 reduction_percent = .70,            # Percentage of the image to keep each time it is clipped
                 max_to_change = 10,                 # Number of times a pixel should be in the highest attention class before clipping
                 plot = False, 
                 v = False):
        
        
        self.device = device
        self.to_tens = transforms.ToTensor()
        self.plot = plot
        self.v = v      
        self.reduction_percent = reduction_percent    
        
        self.threshold_index = 0                     # Number of times the full imagery set has been clipped
        self.threshold_weights = {}                  # Keys = threshold_index (i.e. clip #); Values = Weights of model at point of clip
        self.image_sizes = {}                        # Keys = Municipality ID's; Values = List of iamgery dimensions from each clip
        
        self.epoch = 0
        self.loss = 0
        self.running_train_loss = 0
        self.running_val_loss = 0
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0             
        
        # Variables for the attention based spatial filtering model
        self.stage = 'attn'
        self.attn_model = model
        self.model = model
        self.criterion = criterion
        self.reset_optimizer = optimizer
        self.optimizer = self.reset_optimizer
        
        # Variables for the fully converged model
        self.fc_model = fc_model
        self.fc_optimizer = fc_optimizer
        self.best_loss = 9000000000
        self.num_fc_epochs = num_fc_epochs
        self.fc_batch_size = fc_batch_size

        self.hm_tracker = {}                         # Dictionary to keep track of the attention heatmaps for each municipality
        self.max_to_change = max_to_change           # Number of times a pixel should be in the highest attention class
        self.perc_over_thresh = 0                    # Percentage of images that to cross the max_to_change threshold to run the full imagery clip

        self.scale_estimates = {}                    # Stores the y predictions for each image right before the imagery is clipped
        self.go_dict = {}                            # Dictionary of 0's & 1's; 1 = Image is at convergence dims, 0 = Image is not at convergence dims
        
        
        
    def prep_attn_data(self, image_names, y, split, batch_size):
        
        """
        Function to split data into training and validation dataloaders
        Called from outside of the class beofre model is training
        """
        
        x_train, y_train, x_val, y_val = train_test_split(np.array(image_names), np.array(y), split)
        
        self.x_train, self.y_train, self.x_val, self.y_val = x_train, y_train, x_val, y_val

        train = [(k,v) for k,v in zip(x_train, y_train)]
        val = [(k,v) for k,v in zip(x_val, y_val)]

        train_dl = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
        val_dl = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)
        
        self.set_converence_dims(train_dl, val_dl)
        
        print("Image convergence dimensions: ", self.convergence_dims)

        return train_dl, val_dl        
        
        
        
        
    def set_converence_dims(self, train_dl, val_dl):
        
        """
        Function to find smallest height and width in the full dataset (both training and validation)
        The smallest h & w can come from different images
        All images will converge down to this dimension
        TO_DO: Find a less time-consuming way to do this if possible
        """
        
        smallest_dims = (900000, 900000)
        
        for i,o in train_dl:
            muni_id = i[0].split("/")[3]
            self.go_dict[muni_id] = 0
            
            w, h = load_inputs(i[0]).shape[2], load_inputs(i[0]).shape[3]
            if w < smallest_dims[0]:
                smallest_dims = (w, smallest_dims[1])
            if h < smallest_dims[1]:
                smallest_dims = (smallest_dims[0], h)
            
            
        for i,o in val_dl:
            muni_id = i[0].split("/")[3]
            self.go_dict[muni_id] = 0
            w, h = load_inputs(i[0]).shape[2], load_inputs(i[0]).shape[3]
            if w < smallest_dims[0]:
                smallest_dims = (w, smallest_dims[1])
            if h < smallest_dims[1]:
                smallest_dims = (smallest_dims[0], h)            
            
        self.convergence_dims = smallest_dims
        

        
        
    def update_handler(self, train_dl, val_dl):
        
        """
        General function that ingests the current state of the model and updates what needs to be updated... (:
        """

        # If the stats of the epoch pass the clip check:
        if self.clip_check():
            
            # Update the weights for the image reduction threshold
            self.threshold_weights[self.threshold_index] = deepcopy(self.model.state_dict())
                                    
            # Update the number of times the image has been clipped
            self.threshold_index += 1
            
            # Update the sizes of the images in the dictionary (this happens each time we pass a threshold)
            # and reset the attention heatmap tracker
            self.update_image_sizes(train_dl)
            self.update_image_sizes(val_dl)            
            self.hm_tracker = {}
            
            # Save the y estimates of the data at the current image scale
            self.save_scale_estimates(train_dl, val_dl)

            
            

    def save_scale_estimates(self, train_dl, val_dl):
        
        """
        Function to save the y estimates of the data at the current image scale
        """
        
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
        
        count = 0
        total = len(self.hm_tracker.keys())
        
        # For every attention heatmap, if it's max value is above self.max_to_change, add 1 to the count
        for hm in self.hm_tracker.values():
            if np.max(hm) >= self.max_to_change:
                count += 1
        
        # If there's at least one image with a max value above self.max_to_change, check the percentage
        # of images that have a pixel that fall above this threshold
        if count != 0:
            self.perc_over_thresh = count / total
            if (count / total) >= .50:
                return True
            else:
                return False
        else:
            self.perc_over_thresh = count
            return False
        
                
            
            
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
                self.go_dict[muni_id] = 1
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
            bounds = (self.convergence_dims[0], self.convergence_dims[1])            
            
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
        
        """
        Function to reclassify the attention heatmap into 2 class (highest attention and not highest attention)
        """
        
        # Get the mun_id, the image size and the gradcam and attention heatmap for the current image
        muni_id = impath[0].split("/")[3]
        image_size = (cur_image.shape[2], cur_image.shape[3])
        gradcam, attn_heatmap = get_gradcam(self.model, image_size, cur_image.cuda(), target_layer = self.model.sa)  
        
        # Reclassify everything so the pixels not in the highest attention class = 0 and those in it = 1
        breaks = [i * (255 / 4) for i in range(0, 5)]
        attn_heatmap[(attn_heatmap >= breaks[0]) & (attn_heatmap < breaks[3])] = 0
        attn_heatmap[(attn_heatmap >= breaks[3]) & (attn_heatmap <= breaks[4])] = 1
        
        # Plot the heatmap if so desired
        if self.plot:
            plt.imshow(attn_heatmap)
            plt.savefig(f"{muni_id}_epoch{self.epoch}.png")
            plt.clf()
        
        # If there is no previous heatmap already saved for a given municipality, save the heatmap and move on i.e. if self.epoch == 0...
        if muni_id not in self.hm_tracker:
            self.hm_tracker[muni_id] = attn_heatmap
            return 
        
        # If a previous heatmap does exist add the current heatmap to the old one
        else:
            self.hm_tracker[muni_id] += attn_heatmap
            

            
            
    def prep_input(self, impath):
        
        """
        Function to load in an image and clip it to it's most updated size
        TO-DO: SEE IF YOU CAN DO THE CLIPPING WITHOUT DETACHING FROM THE GPU
        """
        
        # Grab the municipality ID and load the image as a tensor
        muni_id = impath[0].split("/")[3]
        image = self.to_tens(Image.open(impath[0]).convert('RGB')).unsqueeze(0)
        
        # If the muni_id is in the image_sizes dictionary, clip it to the right size
        if muni_id in self.image_sizes.keys():
            for size in self.image_sizes[muni_id]:
                image = image.detach().cpu()[:, :, size[0]:size[1], size[2]:size[3]]
            
        # If it's the first attention-based epoch, save the original image size to the self.image_sizes dictionary
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
        
        if self.stage == 'attn':
            self.update_handler(train_dl, val_dl)
            
        print('{:20}{:20}{:20}{:20}{:20}{:20}'.format("Epoch: " + str(self.epoch), str(round(self.epoch_train_loss, 4)), str(round(self.epoch_val_loss, 4)), str(self.threshold_index), str(sum(self.go_dict.values())), str(self.perc_over_thresh)))
            
        if self.stage == 'fc':
            if self.epoch_val_loss < self.best_loss:
                print("Updating best weights!")
                self.best_loss = self.epoch_val_loss
                self.best_weights = deepcopy(self.model.state_dict())

        self.running_train_loss = 0
        self.running_val_loss = 0
        self.epoch += 1

        
        
    
    def train_attn_model(self, train_dl, val_dl):
        
        """
        Function to train the self-attention based spatial filtering part of the model
        When the trehshold index hits 0, the function passes to the fully connected model training
        """
        
        print('{:20}{:20}{:20}{:20}{:20}{:20}'.format("Epoch", "Training Loss", "Validation Loss", "# times clipped", "# images at scale", "% above max"))
        
        # Until all images reach the convergence dimension, train the attension based model
        while sum(self.go_dict.values()) < (len(train_dl) + len(val_dl)):
                        
            for impath, output in train_dl:                
                input = self.prep_input(impath)
                self.stats(impath, input)
                self.train(input, output)
                
            for impath, output in val_dl:
                input = self.prep_input(impath)
                self.val(input, output)

            self.end_epoch(train_dl, val_dl)
            
        print("Switching to fully connected model!")
        self.train_fc_model(train_dl, val_dl)
            
    
    
    
    def train_fc_model(self, train_dl, val_dl, outside_estimator = None):
        
        """
        Function to train the fully converged model on the standard image sizes after the attention based spatial filtering model has finishing filtering down each of the images
        """
                
        train = [(k,v) for k,v in zip(self.x_train, self.y_train)]
        val = [(k,v) for k,v in zip(self.x_val, self.y_val)]

        train_dl = torch.utils.data.DataLoader(train, batch_size = self.fc_batch_size, shuffle = True)
        val_dl = torch.utils.data.DataLoader(val, batch_size = self.fc_batch_size, shuffle = True)  
        
        
        # Reset all of the variables, and initialize the new model
        self.epoch = 0
        self.model = self.fc_model
        self.optimizer = self.fc_optimizer
        self.stage = 'fc'     
        
        for e in range(self.num_fc_epochs):
                        
            for impath, output in train_dl:
                input = self.prep_input(impath)
                self.train(input, output)
                
            for impath, output in val_dl:
                input = self.prep_input(impath)
                self.val(input, output)                
                
            self.end_epoch(train_dl, val_dl)
            
        self.threshold_weights['fc'] = self.best_weights
        

    def plot_attn_map(self, impath):
        
        """
        Function to save the attention maps for a given image at each threshold
        """
        
        muni_id = impath.split("/")[3]
        cur_image = load_inputs(impath)
        folder = f"{muni_id}_attention_maps"
        
        if not os.path.isdir(folder):
            os.mkdir(folder)
            

        for k in self.threshold_weights.keys():

            if k != 'fc':

                model = self.attn_model
                model.load_state_dict(self.threshold_weights[k])
                model.eval()
                IM_SIZE = (cur_image.shape[2], cur_image.shape[3])
                gradcam, attn_heatmap = get_gradcam(model, IM_SIZE, cur_image.cuda(), target_layer = model.sa) 
                cur_image, new_dims = self.clip_input(cur_image, attn_heatmap)

                fname = os.path.join(f"{muni_id}_attention_maps", f"threshold{k}_muni{muni_id}.png") 
                
                plot_gradcam(gradcam)
                plt.savefig(fname)
                plt.clf()
                
                
                
                
                