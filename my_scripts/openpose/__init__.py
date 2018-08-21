"""
This module loads the OpenPose model, pre-processes the input data and gets the corresponding predictions.

The OpenPose pipeline is the following:
1- Scale the input image at different scales and predict them with the network
2- Combine the resulting predictions into one
3- Find the predictions of the joints by performing non-maximum supression
4- Combine the joints and the limbs to find which of them are connected
5- Given the previous information find which joints/limbs belong to the same person
6- Return the joints for all the people found in the image
"""
# Load the parameters used by the module and the network
from .config_reader import config_reader
param_, model_ = config_reader()

import os
import numpy as np
import torch
from torch.autograd import Variable

from .constants import *
from .model.openpose import OpenPose
from .util import plot_joints
from .prediction import get_prediction, get_prediction_justmodel
from .lin_int_prog import *

openpose_weight_path = os.path.join(os.path.dirname(__file__), 'model', model_['model'])
model_type = model_['type']
if 'COCO' in model_type:
    np_hm = 18
    limbSeq = limbSeq_COCO
    colors = colors_COCO
    nb_limbs = 17
if 'MPI' in model_type:
    np_hm = 15
    limbSeq = limbSeq_MPI
    colors = colors_MPI
    nb_limbs = 14

# To make sure that the timings are correct we load the model into the GPU memory by feeding it a random image.
# To ensure that this is automatically done when importing the module we put it in the __init__.py of the module
# outside any function to ensure that it is called at import time
model = OpenPose(model_type)
model.load_state_dict(torch.load(openpose_weight_path))
model.cuda()
model.float()
model.eval()
model(Variable(torch.from_numpy(np.zeros([1, 3, model_['boxsize'], model_['boxsize']], dtype='float32'))).cuda())


def run(image, scales):
    image_height = image.shape[0]
    # Return the final heatmap and paf (part-affinity field [limb heatmap]) from th network
    heatmap_avg, paf_avg = get_prediction(model, image, model_, scales, model_type)
    # Find all of the joint positions
    all_peaks = count_peaks(heatmap_avg, param_['thre1'])
    # Find all of the limbs that make sense given the joints
    connection_all, special_k = compute_connections(paf_avg, all_peaks, param_['thre2'], image_height, model_type)
    # Combine the joint and limbs to assign them to the different people in the image
    poses = compute_poses(all_peaks, connection_all, special_k, model_type)
    # Return the poses and the final heatmap for the following step

    return poses, heatmap_avg

#sena was here
def run_only_model(image, scales):

    heatmap_avg, heatmaps_scales = get_prediction_justmodel(model, image, model_, scales, model_type)
    poses = torch.zeros((2, heatmap_avg.shape[0]-1))
    poses_scales = torch.zeros((heatmaps_scales.shape[0], 2,heatmap_avg.shape[0]-1))
    for heatmap in range(0, heatmap_avg.shape[0]-1):
        temp = heatmap_avg[heatmap, :, :]
        tempPoses = np.unravel_index(np.argmax(temp), temp.shape)
        poses[:, heatmap] = torch.FloatTensor([tempPoses[1], tempPoses[0]])    

    for heatmap_ind in range(0, heatmaps_scales.shape[0]):
        heatmap_temp = heatmaps_scales[heatmap_ind, :, :]
        for heatmap in range(0, heatmap_avg.shape[0]-1):
            temp = heatmap_temp[heatmap, :, :]
            tempPoses = np.unravel_index(np.argmax(temp), temp.shape)
            poses_scales[heatmap_ind, :, heatmap] = torch.FloatTensor([tempPoses[1], tempPoses[0]])    

    return poses, heatmap_avg, heatmaps_scales, poses_scales
