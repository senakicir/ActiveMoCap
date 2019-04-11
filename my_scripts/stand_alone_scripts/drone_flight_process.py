# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
import sys
sys.path.append("..")

from math import radians, cos, sin, acos
from helpers import rearrange_bones_to_mpi, bones_mpi
from determine_positions import find_2d_pose_openpose

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
    poses, _, _, _ = find_2d_pose_openpose(input_image, scales)
    num_of_joints = poses.shape[0]
    f_pose_2d_str = ""
    for i in range(num_of_joints):
        f_pose_2d_str += str(poses[0,i]) + '\t' + str(poses[1,i]) + '\t' 
    f_pose_2d.write(f_pose_2d_str + "\n")

def record_lift_poses(f_pose_lift):
    determine_relative_3d_pose(mode_lift, current_state, bone_2d, cropped_image, heatmap_2d)
    f_pose_2d_str = ""
    for i in range(num_of_joints):
        f_pose_2d_str += str(poses[0,i]) + '\t' + str(poses[1,i]) + '\t' + str(poses[2,i]) + '\t' 
    f_pose_2d.write(f_pose_2d_str + "\n")

def drone_flight_process(camera_info_xml_file_dir, folders):
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = CameraHandler(folders["f_drone_pos"])
    parser.setContentHandler(handler)
    parser.parse(camera_info_xml_file_dir)

    label_list = handler.label_list
    for label in label_list:
        photo_loc = folders["input_image_dir"]+"/"+label+".jpg"
        image = cv.imread(photo_loc)

        record_2d_poses(folders["f_pose_2d"], image, scales)
        record_lift_poses(folders["f_pose_lift"])

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
    camera_info_xml_file_dir= "/Users/kicirogl/Documents/Drone_Project_Docs/drone_recording/2019_02_isinsu/video_1_full_framerate_2/1/doc.xml"
    input_image_dir = "/Users/kicirogl/Documents/Drone_Project_Docs/drone_recording/2019_02_isinsu/video_1_full_framerate_2"
    output_folder_dir = "/Users/kicirogl/Documents/drone_flight_dataset"

    folders = {"input_image_dir": input_image_dir, "f_drone_pos": open(output_folder_dir + "/drone_pos.txt", 'w'), "f_groundtruth":open(output_folder_dir + "/groundtruth.txt", 'w'), "f_pose_2d":open(output_folder_dir + "/pose_2d.txt", 'w'), "f_pose_lift":open(output_folder_dir + "/pose_lift.txt", 'w')}

    drone_flight_process(camera_info_xml_file_dir, folders)
  
