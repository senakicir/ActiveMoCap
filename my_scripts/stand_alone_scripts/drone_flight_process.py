# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
import sys
sys.path.append("..")

from math import radians, cos, sin, acos
from helpers import rearrange_bones_to_mpi, bones_mpi, numpy_to_tuples
from determine_positions import find_2d_pose_openpose, find_lifted_pose
from crop import Crop, SimpleCrop

import os
import xml.sax
import cv2 as cv

import openpose as openpose_module
import liftnet as liftnet_module

class CameraHandler( xml.sax.ContentHandler ):
    def __init__(self, f_drone_pos):
        self.camera_id = ""
        self.label = ""
        self.transform = ""
        #self.rotation_cov = ""
        #self.location_cov = ""

        self.f_drone_pos = f_drone_pos
        self.f_drone_pos_str = ""

        self.label_list = []

   # Call when an element starts
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "camera":
            print("*****Camera*****")
            self.camera_id = attributes["id"]
            self.label = attributes["label"]    
            print("Camera id:", self.camera_id, ", Label:", self.label)
            self.f_drone_pos_str += self.label + '\t'
            self.label_list.append(self.label)

   # Call when an elements ends
    def endElement(self, tag):
        if self.CurrentData == "transform":
            print("Transform:", self.transform)
            transform_list = self.transform.split(" ")
            for transform_val in transform_list:
                self.f_drone_pos_str += transform_val + '\t'
            self.f_drone_pos.write(self.f_drone_pos_str + "\n")

        self.CurrentData = ""
        self.f_drone_pos_str = ""

   # Call when a character is read
    def characters(self, content):
        if self.CurrentData == "transform":
            self.transform = content

def record_2d_poses(f_pose_2d, input_image, scales):
    pose_2d, heatmap_2d, _, _ = find_2d_pose_openpose(input_image, scales)
    num_of_joints = pose_2d.shape[1]
    f_pose_2d_str = ""
    for i in range(num_of_joints):
        f_pose_2d_str += str(pose_2d[0,i].item()) + '\t' + str(pose_2d[1,i].item()) + '\t' 
    f_pose_2d.write(f_pose_2d_str + "\n")
    return pose_2d, heatmap_2d

def record_lift_poses(f_pose_lift, pose_2d, input_image, heatmap_2d):
    lifted_pose = find_lifted_pose(pose_2d, input_image, heatmap_2d)
    #missing stuff here

    num_of_joints = lifted_pose.shape[1]
    f_pose_lift_str = ""
    for i in range(num_of_joints):
        f_pose_lift_str += str(lifted_pose[0,i].item()) + '\t' + str(lifted_pose[1,i].item()) + '\t' + str(lifted_pose[2,i].item()) + '\t' 
    f_pose_lift.write(f_pose_lift_str + "\n")
    return lifted_pose

def drone_flight_process(camera_info_xml_file_dir, folders):
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = CameraHandler(folders["f_drone_pos"])
    parser.setContentHandler(handler)
    parser.parse(camera_info_xml_file_dir)
    print("Done reading xml")

    label_list = handler.label_list
    for label in label_list:
        print("preparing image", label + ".jpg")
        photo_loc = folders["input_image_dir"]+"/"+label+".jpg"
        image = cv.imread(photo_loc)
        #just to localize the person
        pose_2d, heatmap_2d, _, _ = find_2d_pose_openpose(image, scales=[1,])
        cropper = SimpleCrop(numpy_to_tuples(pose_2d))
        cropped_image = cropper.crop_function(image)

        #find 2d pose and lift pose
        pose_2d, heatmap_2d = record_2d_poses(folders["f_pose_2d"], cropped_image, scales=[0.75, 1, 1.25, 1.5,])
        lifted_pose = record_lift_poses(folders["f_pose_lift"], pose_2d, cropped_image, heatmap_2d)

def display_pose(pose):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111,  projection='3d')
    X = pose[0,:]
    Y = pose[1,:]
    Z = -pose[2,:]
    for _, bone in enumerate(bones_mpi):
        ax.plot(X[bone], Y[bone], Z[bone], c='xkcd:black')

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #plt.show()
    plt.close(fig)


if __name__ == "__main__":
    #camera_info_xml_file_dir = "/Users/kicirogl/Documents/Drone_Project_Docs/drone_recording/2019_02_isinsu/video_1_full_framerate_2/1/doc.xml"
    #input_image_dir = "/Users/kicirogl/Documents/Drone_Project_Docs/drone_recording/2019_02_isinsu/video_1_full_framerate_2"
    #output_folder_dir = "/Users/kicirogl/Documents/drone_flight_dataset"

    camera_info_xml_file_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2/camera_calib.files/0/doc.xml"
    input_image_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2"
    output_folder_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2/drone_flight_dataset"

    folders = {"input_image_dir": input_image_dir, "f_drone_pos": open(output_folder_dir + "/drone_pos.txt", 'w'), "f_groundtruth":open(output_folder_dir + "/groundtruth.txt", 'w'), "f_pose_2d":open(output_folder_dir + "/pose_2d.txt", 'w'), "f_pose_lift":open(output_folder_dir + "/pose_lift.txt", 'w')}

    drone_flight_process(camera_info_xml_file_dir, folders)
  
