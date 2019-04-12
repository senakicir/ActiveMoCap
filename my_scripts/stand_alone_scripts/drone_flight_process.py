# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
import sys
sys.path.append("..")

from math import radians, cos, sin, acos
from helpers import rearrange_bones_to_mpi, bones_mpi, numpy_to_tuples, split_bone_connections
from determine_positions import find_2d_pose_openpose, find_lifted_pose
from crop import Crop, SimpleCrop
from PoseEstimationClient import PoseEstimationClient

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
        self.transform_matrix_list = []

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
            transform_matrix = np.array((transform_list))
            transform_matrix = np.reshape(transform_matrix, (4,4))
            for transform_val in transform_list:
                self.f_drone_pos_str += transform_val + '\t'
            self.f_drone_pos.write(self.f_drone_pos_str + "\n")
            self.transform_matrix_list.append(transform_matrix)

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

def drone_flight_process(energy_parameters, camera_info_xml_file_dir, folders):
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = CameraHandler(folders["f_drone_pos"])
    parser.setContentHandler(handler)
    parser.parse(camera_info_xml_file_dir)
    print("Done reading xml")

    label_list, transform_matrix_list = handler.label_list, handler.transform_matrix_list
    pose_client = PoseEstimationClient(param=energy_parameters, cropping_tool=None, animation="")

    for label_ind, label in enumerate(label_list):
        transform_matrix = transform_matrix_list[label_ind]
        print("preparing image", label + ".jpg")
        photo_loc = folders["input_image_dir"]+"/"+label+".jpg"
        image = cv.imread(photo_loc)

        #just to localize the person
        pose_2d, heatmap_2d, _, _ = find_2d_pose_openpose(image, scales=[1,])
        cropper = SimpleCrop(numpy_to_tuples(pose_2d))
        cropped_image = cropper.crop_function(image)

        #find 2d pose 
        pose_2d, heatmap_2d = record_2d_poses(folders["f_pose_2d"], cropped_image, scales=[0.75, 1, 1.25, 1.5,])
        save_superimposed_openpose(cropped_image, pose_2d.numpy(), folders["output_image_dir"], label)

        #find lift pose
        lifted_pose = record_lift_poses(folders["f_pose_lift"], pose_2d, cropped_image, heatmap_2d)
        save_pose(lifted_pose.cpu().numpy(), folders["output_image_dir"], label)

        #add new frame to pose client 

    #optimize


def save_pose(pose, plot_loc, label):
    file_name = plot_loc + "/lift_pose_" + label + ".jpg"

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

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_superimposed_openpose(image, pose, plot_loc, label):
    file_name = plot_loc + "/openpose_" + label + ".jpg"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)

    left_bone_connections, right_bone_connections, middle_bone_connections = split_bone_connections(bones_mpi)
    for i, bone in enumerate(left_bone_connections):    
        p1, = ax.plot(pose[0, bone], pose[1,bone], color = "r", linewidth=1, label="OpenPose Left")   
    for i, bone in enumerate(right_bone_connections):    
        p2, = ax.plot( pose[0, bone], pose[1,bone], color = "b", linewidth=1, label="OpenPose Right")   
    for i, bone in enumerate(middle_bone_connections):    
        ax.plot( pose[0, bone], pose[1,bone], color = "b", linewidth=1) 
    
    plot_handles = [p1,p2]
    plt.legend(handles=plot_handles)

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    #camera_info_xml_file_dir = "/Users/kicirogl/Documents/Drone_Project_Docs/drone_recording/2019_02_isinsu/video_1_full_framerate_2/1/doc.xml"
    #input_image_dir = "/Users/kicirogl/Documents/Drone_Project_Docs/drone_recording/2019_02_isinsu/video_1_full_framerate_2"
    #output_folder_dir = "/Users/kicirogl/Documents/drone_flight_dataset"

    camera_info_xml_file_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2/camera_calib.files/0/doc.xml"
    input_image_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2"
    output_folder_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2/drone_flight_dataset"
    output_image_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2/drone_flight_dataset/images"
    


    minmax = True #True-min, False-max
    SEED_LIST = [41]#, 5, 2, 12, 1995]
    WOBBLE_FREQ_LIST = [0.5, 1, 2, 5, 20]
    UPDOWN_LIM_LIST = [[-3, -1]]
    LOOKAHEAD_LIST = [0.3]
    go_distance = 3
    upper_lim = -3
    lower_lim = -1 #-2.5

    param_read_M = False
    param_find_M = False
    is_quiet = False
    
    online_window_size = 6
    calibration_length = 0
    calibration_window_size = 200

    precalibration_length = 0
    init_pose_with_gt = True
    find_best_traj = True
    noise_2d_std = 3
    predefined_traj_len = 0

    use_symmetry_term = True
    use_single_joint = False
    #smoothness_mode: 0-velocity, 1-position, 2-all connected, 3-onlyveloconnected, 4-none
    smoothness_mode = 0
    #use_bone_term = True
    #use_lift_term = False
    use_trajectory_basis = False
    num_of_trajectory_param = 5
    num_of_noise_trials = 30
    pose_noise_3d_std = 0.1

    energy_parameters = {"ONLINE_WINDOW_SIZE": 0, "CALIBRATION_WINDOW_SIZE": 1000, "CALIBRATION_LENGTH": 1000, "PRECALIBRATION_LENGTH": 0, "PARAM_FIND_M": True, "PARAM_READ_M": False, "QUIET": False, "MODES": 0, "MODEL": "mpi", "METHOD": "trf", "FTOL": 1e-3, "WEIGHTS": 0, "INIT_POSE_WITH_GT": 0, "NOISE_2D_STD": 0, "USE_SYMMETRY_TERM": True, "USE_SINGLE_JOINT": 0, "SMOOTHNESS_MODE": 0, "USE_TRAJECTORY_BASIS": False, "NUMBER_OF_TRAJ_PARAM": 0, "USE_LIFT_TERM": False, "USE_BONE_TERM": False, "SEED": 0}
    folders = {"input_image_dir": input_image_dir, "output_image_dir": output_image_dir, "f_drone_pos": open(output_folder_dir + "/drone_pos.txt", 'w'), "f_groundtruth":open(output_folder_dir + "/groundtruth.txt", 'w'), "f_pose_2d":open(output_folder_dir + "/pose_2d.txt", 'w'), "f_pose_lift":open(output_folder_dir + "/pose_lift.txt", 'w')}

    drone_flight_process(energy_parameters, camera_info_xml_file_dir, folders)
  
