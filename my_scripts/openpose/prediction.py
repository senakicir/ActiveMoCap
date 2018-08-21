from time import time
import numpy as np
import torch
import torch.nn as nn
from .util import preapreImage


def get_prediction(model, oriImg, model_dict, scale_search, model_type):
    """
    Scales the given input image and feeds it to the model in order to get the predictions.
    The predictions are obtained for each scale given and then are meaned to get the final predictions.
    The final predictions are then returned as numpy arrays. (CPU)

    :param model: The pytorch model that will generate the predictions
    :param oriImg: The input image that will be predicted
    :param model_dict: parameters that are used to prepare the input
    :param scale_search: By what factor to scale the input image. The number of factors determine the number of times the image is passed to the model and the size of it. A scale of 1 generates an image of height 'boxsize'
    :param model_type: Select some parameters given the model that is being used (COCO or MPI)
    :return: The final heatmap and part affinity fileds (limbs)
    """
    if 'COCO' in model_type:
        nb_hm = 19
        nb_paf = 38
    if 'MPI' in model_type:
        nb_hm = 16
        nb_paf = 28

    multiplier = [x * model_dict['boxsize'] / oriImg.shape[0] for x in scale_search]

    heatmap_avg_orig = torch.zeros((len(multiplier), nb_hm, oriImg.shape[0], oriImg.shape[1])).cuda()
    paf_avg_orig = torch.zeros((len(multiplier), nb_paf, oriImg.shape[0], oriImg.shape[1])).cuda()

    # Creating the pyramind and predicting it
    #tic = time()
    times = [None] * len(multiplier)
    for idx, scale in enumerate(multiplier):
        tic_image = time()
        # Scale the image to the desired size
        feed = preapreImage(oriImg, scale, model_dict['stride'], model_dict['padValue'])
        #print "Preparing the image took {}".format(time() - tic_image)


        #tic_model = time()
        output1, output2 = model(feed)
        #openpose_model_eval_time = time() - tic_model
        #times[idx] = openpose_model_eval_time
        #print "Predicting with openpose took {}".format(openpose_model_eval_time)

        heatmap = nn.Upsample((oriImg.shape[0], oriImg.shape[1]), mode='bilinear', align_corners=True).cuda()(output2)
        paf = nn.Upsample((oriImg.shape[0], oriImg.shape[1]), mode='bilinear', align_corners=True).cuda()(output1)

        heatmap_avg_orig[idx] = heatmap[0].data
        paf_avg_orig[idx] = paf[0].data

    #openpose_pyramid_detection_full_time = time() - tic
    ##print "Predicting with openpose the full multi-scale pyramid took {} (just network {})".format(openpose_pyramid_detection_full_time, np.sum(times))
    #print "Openpose took on average {}".format(np.mean(times))
    #print '_' * 80

    #tic = time()
    # Mean all of the prediction to obtain the final one

    heatmap_avg = torch.mean(heatmap_avg_orig, 0).cuda()
    paf_avg = torch.mean(paf_avg_orig, 0).cuda()
    #heatmap_avg = heatmap_avg.cpu().numpy()
    #paf_avg = paf_avg.cpu().numpy()
    #hm_paf_process_time = time() - tic

    #print "Heatmap and PAF processing took {}".format(hm_paf_process_time)
    #print '_' * 80

    # Return the final joint heatmap and part affiny fields
    return heatmap_avg, paf_avg

def get_prediction_justmodel(model, oriImg, model_dict, scale_search, model_type):
    """
    Scales the given input image and feeds it to the model in order to get the predictions.
    The predictions are obtained for each scale given and then are meaned to get the final predictions.
    The final predictions are then returned as numpy arrays. (CPU)

    :param model: The pytorch model that will generate the predictions
    :param oriImg: The input image that will be predicted
    :param model_dict: parameters that are used to prepare the input
    :param scale_search: By what factor to scale the input image. The number of factors determine the number of times the image is passed to the model and the size of it. A scale of 1 generates an image of height 'boxsize'
    :param model_type: Select some parameters given the model that is being used (COCO or MPI)
    :return: The final heatmap and part affinity fileds (limbs)
    """
    if 'COCO' in model_type:
        nb_hm = 19
        nb_paf = 38
    if 'MPI' in model_type:
        nb_hm = 16
        nb_paf = 28

    multiplier = [x * model_dict['boxsize'] / oriImg.shape[0] for x in scale_search]

    heatmap_avg_orig = torch.zeros((len(multiplier), nb_hm, oriImg.shape[0], oriImg.shape[1])).cuda()

    # Creating the pyramind and predicting it
    times = [None] * len(multiplier)
    for idx, scale in enumerate(multiplier):
        feed = preapreImage(oriImg, scale, model_dict['stride'], model_dict['padValue'])
        _, output2 = model(feed)

        heatmap = nn.Upsample((oriImg.shape[0], oriImg.shape[1]), mode='bilinear', align_corners=True).cuda()(output2)
        heatmap_avg_orig[idx] = heatmap[0].data

    # Mean all of the prediction to obtain the final one
    heatmap_avg = torch.mean(heatmap_avg_orig, 0).cuda()
    # Return the final joint heatmap and all the separate heatmaps
    return heatmap_avg, heatmap_avg_orig
