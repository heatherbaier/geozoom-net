# geozoom-net

## Self-attention based spatial filtering

### Experiments:
* Reduction by change: User sets a threshold number of epochs and if the numnber of pixels in the highest attention class hasnt changed by a given percentage in the threshold number of epochs, the imagery is clipped
* Reduction by loss: After the first epoch, the loss if divided by an inputted number of thresholds and each time the validation loss crosses a threshold, the imagery is clipped

### Parameters in both experiments:
1. Convergence dimestion: Images in the dataset will not be clipped smaller than this width and height. Images that are already smaller than this dimension will not be clipped. If an image reaches this dimension, but there are still more training thresholds to meet, it will be skipped during the image clipping portion of the training.
    * Example: convergence_dims = (224,224)
2. Reduction percentage: Percentage by which to reduce the image at each threshold
    * reduction_percent = .70
2. Number of thresholds: Number of times to clip the image (the point at which you cross a threshold is defined byt he experiment itself)
    * num_thresholds = 5
3. Number of fully connected epochs: Number of epochs to run on the imagery after it has been reduced by the attention model
    * num_fc_epochs = 100
4. Plot: If True, the [image, attention heatmap and attention heatmap overlapping the image] are plotted after reaching each threshold
5. Verbose (v): Prints helpful status messages throughout training to aid in debugging
