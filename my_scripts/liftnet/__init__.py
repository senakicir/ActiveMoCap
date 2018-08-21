"""
This module loads the LiftNet model, pre-processes the input data and gets the corresponding predictions.

The LiftNet piepline is the following:
1- Computing the bounding box of the person
2- Cropping the heatmap based on the bounding box
3- Cropping the image based on the bounding box
4- Resizing them to the expected size by LiftNet
5- Getting the 3d pose from LiftNet
6- Returning the 3d pose, cropped heatmap and cropped image
"""
# Start by reading the configuration file
from .config_reader import config_reader
param_, model_ = config_reader()
model_type = model_['type']

from time import time
import importlib
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from .util import same_margin_bounding_box, crop_input

mod = importlib.import_module(model_['module'], package=__package__)
m = mod.LiftNetModel(model_)

def run(image, heatmap_body, pose):
    # Transpose the heatmap to ensure order is compatible for resize
    heatmap_body = np.transpose(heatmap_body, (1, 2, 0))
    bbox = same_margin_bounding_box(pose, model_type, model_['marginBox'])

    # Crop the heatmap and image
    channel_ind = np.fromiter(model_['indexHM'], dtype=np.int) 
    cropped_heatmap = crop_input(heatmap_body, bbox, model_['square'], model_['pad'], model_['padValue'], channel_ind)
    cropped_image = crop_input(image, bbox, model_['square'], model_['pad'], model_['padValue'], [0, 1, 2])
    
    # Resize the crop heatmap ande image
    cropped_heatmap_ = cv2.resize(cropped_heatmap, (model_['boxsize'], model_['boxsize']))
    cropped_image_ = cv2.resize(cropped_image, (model_['boxsize'], model_['boxsize']))

    # Transpose to use it for prediction
    cropped_heatmap = np.transpose(cropped_heatmap_[np.newaxis, ...], (0, 3, 1, 2))
    cropped_image = np.transpose(cropped_image_[np.newaxis, ...], (0, 3, 1, 2))

    # Predict the 3d pose
    threeD_pose, = m.forward(image=Variable(torch.from_numpy(cropped_image).cuda()), heatmap=Variable(torch.from_numpy(cropped_heatmap).cuda()))
    
    return threeD_pose.data, cropped_image_, cropped_heatmap_
