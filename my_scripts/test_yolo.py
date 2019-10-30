import cv2 as cv
import os
import numpy as np

import sys
import darknet as darknet_module

import openpose as openpose_module
import liftnet as liftnet_module

import time as time
import pdb
from crop import Crop, Basic_Crop
from helpers import bones_mpi

import matplotlib.pyplot as plt
from matplotlib import cm, colors

if __name__ == "__main__":
    img_path = ("/cvlabdata2/home/kicirogl/ActiveDrone/simulation_results/2019-10-30-10-50/02_01/0/images/img_0.png")
    im = cv.imread(img_path)

    start1 = time.time()
    predictions = darknet_module.detect(img_path)
    end1 = time.time()

    max_confidence = -1
    bounding_box = None
    for prediction in predictions:
        confidence = prediction[1]
        if (prediction[0] == b'person') and confidence>max_confidence:
            max_confidence = confidence
            bounding_box = prediction[2]

    cropping_tool=Basic_Crop (margin=0.2)
    cropping_tool.update_bbox(bounding_box)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    bbox_corners_x, bbox_corners_y = cropping_tool.return_bbox_coord()
    plt.plot(bbox_corners_x, bbox_corners_y)
    plt.savefig("/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/yolo.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cropped_im = cropping_tool.crop_image(im)
    ax.imshow(cropped_im)
    plt.savefig("/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/basic_crop.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    poses, heatmaps, heatmaps_scales, poses_scales = openpose_module.run_only_model(cropped_im, [0.5,1,1.5])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(cropped_im)
    for i, bone in enumerate(bones_mpi):
        p0, = ax.plot(poses[0, bone], poses[1,bone], color = "r", linewidth=2)
    plt.savefig("/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/basic_crop_openpose.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)


    uncropped_pose = cropping_tool.uncrop_pose(poses)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    for i, bone in enumerate(bones_mpi):
        p0, = ax.plot(uncropped_pose[0, bone], uncropped_pose[1,bone], color = "r", linewidth=2)
    plt.savefig("/cvlabdata2/home/kicirogl/ActiveDrone/my_scripts/openpose.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)