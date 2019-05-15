# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
import sys
sys.path.append("..")

from math import radians, cos, sin, acos, degrees, pi
from helpers import rearrange_bones_to_mpi, bones_mpi, numpy_to_tuples, split_bone_connections, return_lift_bone_connections, euler_to_rotation_matrix, rotation_matrix_to_euler
from determine_positions import find_2d_pose_openpose, find_lifted_pose
from crop import Crop, SimpleCrop
from PoseEstimationClient import PoseEstimationClient
from Lift_Client import calculate_bone_directions
from pose3d_optimizer_scipy import pose3d_calibration_parallel_wrapper
from scipy.optimize import least_squares
import time as time
import os

import os
import xml.sax
import cv2 as cv

import openpose as openpose_module
import liftnet as liftnet_module
import pdb
import random

cherry_pick_frames = ["frame0030.jpg", "frame0040.jpg", "frame0115.jpg", "frame0150.jpg", "frame0155.jpg",
                        "frame0160.jpg", "frame0175.jpg", "frame0200.jpg", "frame0210.jpg", "frame0260.jpg", "frame0265.jpg", 
                        "frame0270.jpg", "frame0274.jpg", "frame0380.jpg", "frame0390.jpg", "frame0395.jpg", "frame0415.jpg", 
                        "frame0473.jpg", "frame0509.jpg", "frame0510.jpg", "frame0514.jpg",
                        "frame0517.jpg", "frame0523.jpg", "frame0527.jpg", "frame0530.jpg", "frame0534.jpg", "frame0537.jpg",
                        "frame0541.jpg", "frame0544.jpg", "frame0549.jpg", "frame0553.jpg", "frame0557.jpg", "frame0564.jpg",
                        "frame0575.jpg", "frame0580.jpg", "frame0585.jpg", "frame0591.jpg", "frame0597.jpg", "frame0604.jpg",
                        "frame0608.jpg", "frame0619.jpg", "frame0623.jpg", "frame0625.jpg", "frame0628.jpg", "frame0630.jpg",
                        "frame0633.jpg", "frame0636.jpg", "frame0639.jpg", "frame0643.jpg", "frame0646.jpg", "frame0650.jpg",  "frame0655.jpg"]

class CameraHandler( xml.sax.ContentHandler ):
    def __init__(self, f_drone_pos, f_intrinsics):
        self.camera_id = ""
        self.label = ""
        self.transform = ""
        #self.rotation_cov = ""
        #self.location_cov = ""

        self.f_drone_pos = f_drone_pos
        self.f_drone_pos_str = ""
        self.f_intrinsics = f_intrinsics

        self.label_list = []
        self.transform_matrix_list = []

        #intrinsics
        self.f = 0
        self.size_x = 0
        self.size_y = 0
        self.cx = 0
        self.cy = 0
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0 
        self.p1 = 0 
        self.p2 = 0 

   # Call when an element starts
    def startElement(self, tag, attributes):
        self.CurrentData = tag            
        if tag == "camera":
            self.camera_id = attributes["id"]
            self.label = attributes["label"]    
            self.f_drone_pos_str += self.label + '\t'

        if tag == "resolution":
            self.size_x = int(attributes["width"])
            self.size_y = int(attributes["height"])

   # Call when an elements ends
    def endElement(self, tag):
        if self.CurrentData == "cy":
            #print("Saving intrinsics now")
            f_intrinsics_str = str(self.f) + '\t' + str(self.cx) + '\t' + str(self.cy) + '\t' + str(self.size_x) + '\t' + str(self.size_y)
            self.f_intrinsics.write(f_intrinsics_str + "\n")

        if self.CurrentData == "transform":
            print_stuff = False
            if print_stuff:
                print("*****Camera*****")
                print("Camera id:", self.camera_id, ", Label:", self.label)
                print("Transform:", self.transform)
            transform_list = self.transform.split(" ")
            transform_matrix_np = np.reshape(np.array(transform_list, dtype=float), newshape = [4,4], order = "C")
            transform_matrix = torch.from_numpy(transform_matrix_np).float()
            for transform_val in transform_list:
                self.f_drone_pos_str += transform_val + '\t'
            self.f_drone_pos.write(self.f_drone_pos_str + "\n")
            self.transform_matrix_list.append(transform_matrix)
            self.label_list.append(self.label)

        self.CurrentData = ""
        self.f_drone_pos_str = ""

   # Call when a character is read
    def characters(self, content):
        if self.CurrentData == "transform":
            self.transform = content
        if self.CurrentData  == "fx":
            print("focal length is", content)
            self.f = float(content)
        if self.CurrentData  == "cx":
            self.cx = float(content)
        if self.CurrentData  == "cy":
            self.cy = float(content)

def find_2d_pose(input_image, scales, output_image_dir, label):
    cropped_pose_2d, heatmap_2d, _, _ = find_2d_pose_openpose(input_image, scales=scales)
    cropper = SimpleCrop(numpy_to_tuples(cropped_pose_2d))
    cropped_image = cropper.crop_function(input_image)

    cropped_pose_2d, heatmap_2d, _, _ = find_2d_pose_openpose(cropped_image, scales=scales)
    vis_superimposed_openpose(cropped_image, cropped_pose_2d.numpy(), output_image_dir, "openpose_cropped_", label)
    pose_2d = cropper.uncrop_pose(cropped_pose_2d)
    vis_superimposed_openpose(input_image, pose_2d.numpy(), output_image_dir, "openpose", label)

    return pose_2d, cropped_pose_2d

def find_lift_pose(pose_2d, projection_client, transformation_matrix, input_image, heatmap_2d):
    lifted_pose = find_lifted_pose(pose_2d, input_image, heatmap_2d)
    pose3d_relative = projection_client.camera_to_world(lifted_pose.cpu(), transformation_matrix)
    return pose3d_relative

def pose_3d_estimate(pose_client, use_these_ind, transform_matrix_tensor, inv_transform_matrix_tensor, pose_2d_tensor):
    _, _, num_of_joints, _ = pose_client.model_settings()

    print("Estimating pose using these ind:", use_these_ind, "using", len(use_these_ind), "frames")
    for linecount, label_ind in enumerate(use_these_ind):
        inv_transformation_matrix = inv_transform_matrix_tensor[label_ind, :, :]
        pose_2d =  pose_2d_tensor[label_ind, :, :]
        pose3d_lift_directions =0 

        #add new frame to pose client 
        pose_client.addNewFrame(pose_2d, pose_2d, inv_transformation_matrix, linecount, np.zeros((3,num_of_joints)), pose3d_lift_directions)

    #optimize and find gt 3d pose
    pose_client.set_initial_pose(0, np.zeros((3,num_of_joints)), pose_2d_tensor[use_these_ind[0], :, :], transform_matrix_tensor[use_these_ind[0], :, :])
    pose3d_init_scrambled = pose_client.pose_3d_preoptimization.copy()

    pose3d_init = np.reshape(a = pose3d_init_scrambled, newshape = [pose_client.result_size,], order = "C")
    
    objective = pose3d_calibration_parallel_wrapper()
    objective_jacobian =  objective.jacobian
    objective.reset(pose_client)
    optimized_res = least_squares(objective.forward, pose3d_init, jac=objective_jacobian, bounds=(-np.inf, np.inf), method=pose_client.method, ftol=pose_client.ftol)
    optimized_poses = np.reshape(a = optimized_res.x, newshape =  pose_client.result_shape, order = "C")
    pose_client.update3dPos(optimized_poses)

    gt_3d_pose = pose_client.current_pose
    return gt_3d_pose


def reorient_human_and_drones(gt_3d_pose, transform_matrix_tensor, joint_names):

    human_displacement = torch.zeros([3,1])
    human_displacement[0:2, 0] = torch.from_numpy(gt_3d_pose[0:2, joint_names.index('spine1')]).float()
    if ( gt_3d_pose[2, joint_names.index('left_foot')] < gt_3d_pose[2, joint_names.index('right_foot')]):
        human_displacement[2,0] = gt_3d_pose[2, joint_names.index('left_foot')]
    else:
        human_displacement[2,0] = gt_3d_pose[2, joint_names.index('right_foot')]
    gt_3d_pose = gt_3d_pose - human_displacement.numpy()
    for i in range(transform_matrix_tensor.shape[0]):
        transform_matrix_tensor[i, 0:3, 3] = transform_matrix_tensor[i, 0:3, 3] - human_displacement[:,0]  
    
    spine_vector = gt_3d_pose[:, joint_names.index('neck')] - gt_3d_pose[:, joint_names.index('spine1')] 
    shoulder_vector = gt_3d_pose[:, joint_names.index('left_arm')] - gt_3d_pose[:, joint_names.index('right_arm')] 
    normal_vec = np.cross(spine_vector, shoulder_vector)
   
    normal_vec_unit = normal_vec/np.linalg.norm(normal_vec)
    spine_vector_unit = spine_vector/np.linalg.norm(spine_vector)
    shoulder_vector = np.cross(normal_vec_unit, spine_vector_unit)
    shoulder_vector_unit = shoulder_vector/np.linalg.norm(shoulder_vector)

    R = torch.from_numpy(np.concatenate([normal_vec_unit[:, np.newaxis], shoulder_vector_unit[:, np.newaxis], spine_vector_unit[:, np.newaxis]], axis=1)).float()
    overall_transformation_matrix = torch.cat((torch.cat((R, torch.zeros([3,1])), dim=1), torch.FloatTensor([[0,0,0,1]])), dim=0)
    inv_overall_transformation_matrix = torch.inverse(overall_transformation_matrix)

    reoriented_3d_pose = np.dot(inv_overall_transformation_matrix.numpy(), np.concatenate((gt_3d_pose, np.ones([1,gt_3d_pose.shape[1]])), axis=0))

    reoriented_transform_matrix = torch.zeros(transform_matrix_tensor.shape)
    reoriented_inv_transform_matrix = torch.zeros(transform_matrix_tensor.shape)
    for i in range(transform_matrix_tensor.shape[0]):
        reoriented_transform_matrix[i, :, :] = torch.mm(inv_overall_transformation_matrix, transform_matrix_tensor[i, :, :])
        reoriented_inv_transform_matrix[i, :, :] = torch.inverse(reoriented_transform_matrix[i, :, :])
    return reoriented_3d_pose, reoriented_transform_matrix, reoriented_inv_transform_matrix

def prune_indices(transform_matrix_tensor, label_list, use_these_ind, distance_tol=1):
    pruned_ind_list = use_these_ind.copy()
    for i in pruned_ind_list:
        curr_drone_displacement = transform_matrix_tensor[i, 0:3, 3]
        j_ind = 0
        while j_ind < len(pruned_ind_list):
        #for j in pruned_ind_list:
            j = pruned_ind_list[j_ind]
            if i != j:
                other_drone_displacement =  transform_matrix_tensor[j, 0:3, 3]
                if (np.linalg.norm(curr_drone_displacement-other_drone_displacement) < distance_tol):
                    pruned_ind_list.remove(j)
                    j_ind -=1
            j_ind +=1

    pruned_label_list = []
    for i in pruned_ind_list:
        pruned_label_list.append(label_list[i])
    
    print(pruned_ind_list)
    return pruned_ind_list, pruned_label_list

def find_3d_gt(energy_parameters, filenames, mode):
    files = {"f_drone_pos": open(filenames["f_drone_pos"], "r"),
            "f_intrinsics":open(filenames["f_intrinsics"], "r"),
            "f_pose_2d":open(filenames["f_pose_2d"], "r"),
            "f_pose_lift":open(filenames["f_pose_lift"], "r"),
            "f_groundtruth": open(filenames["f_groundtruth"], "w"),
            "f_groundtruth_reoriented": open(filenames["f_groundtruth_reoriented"], "w"),
            "f_drone_pos_reoriented": open(filenames["f_drone_pos_reoriented"], "w")}

    label_list, transform_matrix_tensor, inverse_transform_matrix_tensor = read_transformation_matrix(files["f_drone_pos"])
    focal_length, cx, cy = read_intrinsics(files["f_intrinsics"])

    pose_client = PoseEstimationClient(param=energy_parameters, simulation_mode="drone_flight_data", cropping_tool=None, animation="", intrinsics_focal=focal_length, intrinsics_px=cx, intrinsics_py=cy)
    bone_connections, joint_names, num_of_joints, hip_index = pose_client.model_settings()

    pose_2d_tensor = read_pose_from_file(files["f_pose_2d"], 2, num_of_joints)

    if mode == "cherry-pick":
        good_frames =  cherry_pick_frames
        use_these_ind = [label_list.index(frame) for frame in good_frames]
    elif mode == "all":
        use_these_ind = list(range(len(label_list))) 
    elif mode == "ransac":
        use_these_ind = run_ransac(pose_client, pose_2d_tensor, transform_matrix_tensor, inverse_transform_matrix_tensor, label_list)

    gt_3d_pose = pose_3d_estimate(pose_client, use_these_ind, transform_matrix_tensor, inverse_transform_matrix_tensor, pose_2d_tensor)
    vis_pose(gt_3d_pose, filenames["gt_folder_dir"], '', 'gt_3d_pose')   
    plot_all_drone_pos(transform_matrix_tensor[use_these_ind, :, :], gt_3d_pose, filenames["gt_folder_dir"], "", "general_plot", use_these_ind)
    record_3d_poses(files["f_groundtruth"], gt_3d_pose)

    #####
    gt_3d_pose, transform_matrix_tensor, inverse_transform_matrix_tensor = reorient_human_and_drones(gt_3d_pose, transform_matrix_tensor, joint_names)
    plot_all_drone_pos(transform_matrix_tensor[use_these_ind,:,:], gt_3d_pose, filenames["gt_folder_dir"], "", "general_plot_reoriented", use_these_ind)

    pruned_ind, pruned_label_list = prune_indices(transform_matrix_tensor, label_list, use_these_ind, distance_tol=3.5)
    print("Number of pruned ind", len(pruned_ind))

    vis_pose(gt_3d_pose, filenames["gt_folder_dir"], '', 'gt_3d_pose_reoriented')   
    plot_all_drone_pos(transform_matrix_tensor[pruned_ind,:,:], gt_3d_pose, filenames["gt_folder_dir"], "", "general_plot_reoriented_pruned", pruned_ind)
    
    record_3d_poses(files["f_groundtruth_reoriented"], gt_3d_pose)
    record_drone_transformation_matrix(files["f_drone_pos_reoriented"], transform_matrix_tensor[pruned_ind,:,:], pruned_label_list)

def drone_flight_process_all_data(filenames, energy_parameters):
    files = {"f_drone_pos": open(filenames["f_drone_pos"], "w"), 
            "f_intrinsics":open(filenames["f_intrinsics"], "w"),
            "f_pose_2d":open(filenames["f_pose_2d"], "w"),
            "f_pose_lift":open(filenames["f_pose_lift"], "w"), }

    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = CameraHandler(files["f_drone_pos"], files["f_intrinsics"])
    parser.setContentHandler(handler)
    parser.parse(camera_info_xml_file_dir)

    label_list, transform_matrix_list, focal_length, cx, cy = handler.label_list, handler.transform_matrix_list, handler.f, handler.cx, handler.cy 

    pose_client = PoseEstimationClient(param=energy_parameters, simulation_mode="drone_flight_data", cropping_tool=None, animation="", intrinsics_focal=focal_length, intrinsics_px=cx, intrinsics_py=cy)
    bone_connections, joint_names, num_of_joints, hip_index = pose_client.model_settings()

    for label_ind in range(len(label_list)):
        label = label_list[label_ind]
        print("preparing image", label)
        #inv_transformation_matrix = torch.inverse(transformation_matrix)
        photo_loc = filenames["input_image_dir"]+"/"+label
        image = cv.imread(photo_loc)

        #find 2d pose
        pose_2d, _ = find_2d_pose(input_image=image, scales=[0.5, 0.75, 1], output_image_dir=filenames["openpose_liftnet_image_dir"], label=label)
        record_2d_poses(files["f_pose_2d"], pose_2d)

        #find lift pose
        #transformation_matrix = transform_matrix_list[label_ind]
        #lifted_pose = find_lift_pose(cropped_pose_2d, pose_client.projection_client, transformation_matrix, cropped_image, heatmap_2d)
        #record_3d_poses(files["f_pose_lift"], lifted_pose)
        #vis_pose(lifted_pose.cpu().numpy(), filenames["openpose_liftnet_image_dir"], label, 'lift_pose')

def run_ransac(pose_client, pose_list_2d, transform_matrix_tensor, inverse_transform_matrix_tensor, label_list):
    best_err = np.inf
    best_ind_list = []
    full_ind_list = list(range(len(label_list)))
    for i in range(20):
        maybe_inlier_label_ind = random.sample(full_ind_list, 10)
        gt_3d_pose = pose_3d_estimate(pose_client, maybe_inlier_label_ind, transform_matrix_tensor, inverse_transform_matrix_tensor, pose_list_2d)
        gt_3d_pose_torch = torch.from_numpy(gt_3d_pose).float()
        also_inliers_label_ind = []
        for label_ind in full_ind_list:
            if label_ind not in maybe_inlier_label_ind:
                inv_transformation_matrix = inverse_transform_matrix_tensor[label_ind, :, :] 
                pose_2d_actual = pose_list_2d[label_ind, :, :]
                pose_2d_estimate = pose_client.projection_client.take_single_projection(gt_3d_pose_torch, inv_transformation_matrix)
                maybe_err = np.linalg.norm(pose_2d_estimate-pose_2d_actual)/pose_2d_actual.shape[1]
                if maybe_err < 20:
                    also_inliers_label_ind.append(label_ind)
        
        if len(also_inliers_label_ind) > 20:
            better_inlier_list = maybe_inlier_label_ind + also_inliers_label_ind
            better_3d_pose = pose_3d_estimate(pose_client, better_inlier_list, transform_matrix_tensor, inverse_transform_matrix_tensor, pose_list_2d)
            better_3d_pose_torch = torch.from_numpy(better_3d_pose).float()

            this_err = 0
            for label_ind in better_inlier_list:
                inv_transformation_matrix = inverse_transform_matrix_tensor[label_ind, :, :] 
                pose_2d_actual = pose_list_2d[label_ind, :, :]
                pose_2d_estimate = pose_client.projection_client.take_single_projection(better_3d_pose_torch, inv_transformation_matrix)
                this_err += (np.linalg.norm(pose_2d_estimate-pose_2d_actual)/pose_2d_actual.shape[1] )/len(better_inlier_list)
                           
            if this_err < best_err:
                #best_model = better_3d_pose_torch
                best_err = this_err
                best_ind_list = better_inlier_list.copy()

    return best_ind_list


####### visualization functions and dataloader functions
def vis_pose(pose, plot_loc, label, custom_name):
    file_name = plot_loc + "/" + custom_name  + '_' + label + ".jpg"

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111,  projection='3d')
    X = pose[0,:]
    Y = pose[1,:]
    Z = pose[2,:]
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
    #plt.show()
    plt.close(fig)

def vis_superimposed_openpose(image, pose, plot_loc, custom_name, label):
    file_name = plot_loc + '/' + custom_name + label + ".jpg"

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

def plot_all_drone_pos_debug(transformation_matrix_tensor_old, gt_3d_pose_old, transformation_matrix_tensor_new, gt_3d_pose_new):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121,  projection='3d')
    X = gt_3d_pose_old[0,:].tolist()
    Y = gt_3d_pose_old[1,:].tolist()
    Z = gt_3d_pose_old[2,:].tolist()
    for _, bone in enumerate(bones_mpi):
        ax.plot(gt_3d_pose_old[0,bone], gt_3d_pose_old[1,bone], gt_3d_pose_old[2,bone], c='xkcd:black')

    for ind in range(transformation_matrix_tensor_old.shape[0]):
        transformation_matrix = transformation_matrix_tensor_old[ind, :, :]
        C_drone = transformation_matrix[0:3, 3]
        R_drone = transformation_matrix[0:3, 0:3]
        axis1 = R_drone@torch.t(torch.FloatTensor([[0,0,1]]))
        axis2 = R_drone@torch.t(torch.FloatTensor([[0,1,0]]))
        axis3 = R_drone@torch.t(torch.FloatTensor([[1,0,0]]))

        axis1 = 3*axis1/torch.norm(axis1)
        axis2 = 3*axis2/torch.norm(axis2)
        axis3 = 3*axis3/torch.norm(axis3)

        ax.scatter(C_drone[0], C_drone[1], C_drone[2], c='xkcd:pink')
        ax.plot([C_drone[0], C_drone[0]+axis1[0,0]], [C_drone[1], C_drone[1]+axis1[1,0]], [C_drone[2], C_drone[2]+axis1[2,0]], c='xkcd:red')
        ax.plot([C_drone[0], C_drone[0]+axis2[0,0]], [C_drone[1], C_drone[1]+axis2[1,0]], [C_drone[2], C_drone[2]+axis2[2,0]], c='xkcd:blue')
        ax.plot([C_drone[0], C_drone[0]+axis3[0,0]], [C_drone[1], C_drone[1]+axis3[1,0]], [C_drone[2], C_drone[2]+axis3[2,0]], c='xkcd:green')

        X.append(C_drone[0])
        Y.append(C_drone[1])
        Z.append(C_drone[2])
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=30., azim=135)

    ax = fig.add_subplot(122,  projection='3d')
    X = gt_3d_pose_new[0,:].tolist()
    Y = gt_3d_pose_new[1,:].tolist()
    Z = gt_3d_pose_new[2,:].tolist()
    for _, bone in enumerate(bones_mpi):
        ax.plot(gt_3d_pose_new[0,bone], gt_3d_pose_new[1,bone], gt_3d_pose_new[2,bone], c='xkcd:black')

    for ind in range(transformation_matrix_tensor_new.shape[0]):
        transformation_matrix = transformation_matrix_tensor_new[ind, :, :]
        C_drone = transformation_matrix[0:3, 3]
        R_drone = transformation_matrix[0:3, 0:3]
        axis1 = R_drone@torch.t(torch.FloatTensor([[0,0,1]]))
        axis2 = R_drone@torch.t(torch.FloatTensor([[0,1,0]]))
        axis3 = R_drone@torch.t(torch.FloatTensor([[1,0,0]]))

        axis1 = 3*axis1/torch.norm(axis1)
        axis2 = 3*axis2/torch.norm(axis2)
        axis3 = 3*axis3/torch.norm(axis3)

        ax.scatter(C_drone[0], C_drone[1], C_drone[2], c='xkcd:pink')
        ax.plot([C_drone[0], C_drone[0]+axis1[0,0]], [C_drone[1], C_drone[1]+axis1[1,0]], [C_drone[2], C_drone[2]+axis1[2,0]], c='xkcd:red')
        ax.plot([C_drone[0], C_drone[0]+axis2[0,0]], [C_drone[1], C_drone[1]+axis2[1,0]], [C_drone[2], C_drone[2]+axis2[2,0]], c='xkcd:blue')
        ax.plot([C_drone[0], C_drone[0]+axis3[0,0]], [C_drone[1], C_drone[1]+axis3[1,0]], [C_drone[2], C_drone[2]+axis3[2,0]], c='xkcd:green')

        X.append(C_drone[0])
        Y.append(C_drone[1])
        Z.append(C_drone[2])

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=30., azim=135)
    plt.show()
    plt.close(fig)

def plot_all_drone_pos(transformation_matrix_tensor, gt_3d_pose, plot_loc, label, custom_name, indices):
    file_name = plot_loc + "/" + custom_name  + ".jpg"

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111,  projection='3d')
    X = gt_3d_pose[0,:].tolist()
    Y = gt_3d_pose[1,:].tolist()
    Z = gt_3d_pose[2,:].tolist()
    for _, bone in enumerate(bones_mpi):
        ax.plot(gt_3d_pose[0,bone], gt_3d_pose[1,bone], gt_3d_pose[2,bone], c='xkcd:black')

    for ind in range(transformation_matrix_tensor.shape[0]):
        transformation_matrix = transformation_matrix_tensor[ind, :, :]
        C_drone = transformation_matrix[0:3, 3]
        R_drone = transformation_matrix[0:3, 0:3]
        axis1 = R_drone@torch.t(torch.FloatTensor([[0,0,1]]))
        axis2 = R_drone@torch.t(torch.FloatTensor([[0,1,0]]))
        axis3 = R_drone@torch.t(torch.FloatTensor([[1,0,0]]))

        axis1 = 3*axis1/torch.norm(axis1)
        axis2 = 3*axis2/torch.norm(axis2)
        axis3 = 3*axis3/torch.norm(axis3)

        ax.scatter(C_drone[0], C_drone[1], C_drone[2], c='xkcd:pink')
        ax.plot([C_drone[0], C_drone[0]+axis1[0,0]], [C_drone[1], C_drone[1]+axis1[1,0]], [C_drone[2], C_drone[2]+axis1[2,0]], c='xkcd:red')
        ax.plot([C_drone[0], C_drone[0]+axis2[0,0]], [C_drone[1], C_drone[1]+axis2[1,0]], [C_drone[2], C_drone[2]+axis2[2,0]], c='xkcd:blue')
        ax.plot([C_drone[0], C_drone[0]+axis3[0,0]], [C_drone[1], C_drone[1]+axis3[1,0]], [C_drone[2], C_drone[2]+axis3[2,0]], c='xkcd:green')
        #ax.text(C_drone[0], C_drone[1], C_drone[2], str(indices[ind]))

        X.append(C_drone[0])
        Y.append(C_drone[1])
        Z.append(C_drone[2])

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=30., azim=135)

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def record_2d_poses(f_pose_2d, pose_2d):
    num_of_joints = pose_2d.shape[1]
    f_pose_2d_str = ""
    for i in range(num_of_joints):
        f_pose_2d_str += str(pose_2d[0,i].item()) + '\t' + str(pose_2d[1,i].item()) + '\t' 
    f_pose_2d.write(f_pose_2d_str + "\n")

def record_3d_poses(f_pose, pose_3d):
    num_of_joints = pose_3d.shape[1]
    f_pose_str = ""
    for i in range(num_of_joints):
        f_pose_str += str(pose_3d[0,i].item()) + '\t' + str(pose_3d[1,i].item()) + '\t' + str(pose_3d[2,i].item()) + '\t' 
    f_pose.write(f_pose_str + "\n")

def record_drone_transformation_matrix(f_drone_pos, transformation_matrix, label_list):
    for ind in range(transformation_matrix.shape[0]):
        label = label_list[ind]
        flattened_transformation_matrix = np.reshape(transformation_matrix[ind, :].numpy(), (16, ))
        f_drone_pos_str = str(label) + '\t'
        for i in range (16):
            f_drone_pos_str += str(float(flattened_transformation_matrix[i])) + '\t'
        f_drone_pos.write(f_drone_pos_str + "\n")

#def plot_stuff(filenames):
#    files = {"f_drone_pos": open(filenames["f_drone_pos"], "r"), 
#            "f_groundtruth":open(filenames["f_groundtruth"], "r")}
#    human_gt, transformation_matrix = read_gt_and_transformation_matrix(files)
#    plot_all_drone_pos(transformation_matrix, human_gt, filenames["output_image_dir"], "", "drone_pos")
    

#####reader functions
def read_transformation_matrix(f_drone_pos):
    whole_file = pd.read_csv(f_drone_pos, sep='\t').as_matrix()
    label_list = whole_file[:,0].tolist()
    transformation_matrix = whole_file[:,1:-1].astype('float')
    num_of_files = transformation_matrix.shape[0]
    transformation_matrix_tensor = torch.zeros([num_of_files, 4, 4])
    inv_transformation_matrix_tensor = torch.zeros([num_of_files, 4, 4])
    for ind in range(num_of_files):
        transformation_matrix_tensor[ind, :,:] = torch.from_numpy(np.reshape(transformation_matrix[ind, :], (4,4))).float()
        inv_transformation_matrix_tensor[ind, :,:] = torch.inverse(transformation_matrix_tensor[ind, :,:] )
    return label_list, transformation_matrix_tensor, inv_transformation_matrix_tensor

def read_gt_and_transformation_matrix(files):
    f_groundtruth, f_drone_pos = files["f_groundtruth"], files["f_drone_pos"]
    gt_str = f_groundtruth.read().split('\t')
    gt_str.pop()
    pose3d_list = [float(joint) for joint in gt_str]
    pose_3d_gt = np.reshape(np.array(pose3d_list, dtype=float), newshape=(3,-1), order="F")

    transformation_matrix = pd.read_csv(f_drone_pos, sep='\t').as_matrix()[:,1:-1].astype('float')
    transformation_matrix_list = []
    for ind in range(transformation_matrix.shape[0]):
        transformation_matrix_list.append(np.reshape(transformation_matrix[ind, :], (4,4)))

    return pose_3d_gt, transformation_matrix_list

def read_pose_from_file(input_file, dim, num_of_joints):
    whole_file = pd.read_csv(input_file, sep='\t').as_matrix()[:,:-1].astype('float')
    pose = torch.zeros(whole_file.shape[0], dim, num_of_joints)
    for i in range(whole_file.shape[0]):
        pose[i, :, : ] = torch.from_numpy(np.reshape(whole_file[i, :], newshape=(dim, -1), order="F")).float()
    return pose

def read_intrinsics(f_intrinsics):
    intrinsics_str = f_intrinsics.read().split('\t')
    intrinsics_str.pop()
    intrinsics = [float(value) for value in intrinsics_str]
    return intrinsics[0], intrinsics[1], intrinsics[2]

def close_files(files):
    items = ["f_drone_pos", "f_groundtruth", "f_pose_2d", "f_pose_lift", "f_intrinsics"]
    for a_file in files:
        a_file.close()


if __name__ == "__main__":
    #camera_info_xml_file_dir = "/Users/kicirogl/Documents/Drone_Project_Docs/drone_recording/2019_02_isinsu/video_1_full_framerate_2/1/doc.xml"
    #input_image_dir = "/Users/kicirogl/Documents/Drone_Project_Docs/drone_recording/2019_02_isinsu/video_1_full_framerate_2"
    #output_folder_dir = "/Users/kicirogl/Documents/drone_flight_dataset"
    process_all_data = False
    find_3d_pose = True
    mode = "ransac" #0: 'all', 1: 'cherry-pick', 2: 'ransac'

    date_time_name = time.strftime("%Y-%m-%d-%H-%M")

    main_dir = "/cvlabdata2/home/kicirogl/ActiveDrone/drone_flight/2019_02_isinsu/video_1_full_framerate_2_trial_2"
    camera_info_xml_file_dir = main_dir+ "/2019_02_isinsu_subsampled.files/0/doc.xml"
    input_image_dir = main_dir
    general_output_folder = main_dir + "/drone_flight_dataset/" 
    gt_folder_dir = general_output_folder + date_time_name + "_" + mode + '/'
    openpose_liftnet_image_dir = general_output_folder + "openpose_liftnet_images"


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

    energy_parameters = {"ONLINE_WINDOW_SIZE": 1, "CALIBRATION_WINDOW_SIZE": 1000, "CALIBRATION_LENGTH": 1000, "PRECALIBRATION_LENGTH": 0, "PARAM_FIND_M": True, "PARAM_READ_M": False, "QUIET": False, "MODES": 0, "MODEL": "mpi", "METHOD": "trf", "FTOL": 1e-3, "WEIGHTS": 0, "INIT_POSE_WITH_GT": 0, "NOISE_2D_STD": 0, "USE_SYMMETRY_TERM": True, "USE_SINGLE_JOINT": 0, "SMOOTHNESS_MODE": 0, "USE_TRAJECTORY_BASIS": False, "NUMBER_OF_TRAJ_PARAM": 0, "USE_LIFT_TERM": False, "USE_BONE_TERM": False, "SEED": 0}
    filenames = {"input_image_dir": input_image_dir, 
                "openpose_liftnet_image_dir": openpose_liftnet_image_dir, 
                "gt_folder_dir": gt_folder_dir,
                "camera_info_xml_file_dir": camera_info_xml_file_dir,
                "f_drone_pos": general_output_folder + "drone_pos.txt", 
                "f_drone_pos_reoriented": gt_folder_dir + "drone_pos_reoriented.txt", 
                "f_groundtruth": gt_folder_dir + "groundtruth.txt", 
                "f_groundtruth_reoriented": gt_folder_dir + "groundtruth_reoriented.txt", 
                "f_pose_2d": general_output_folder + "pose_2d.txt", 
                "f_pose_lift": general_output_folder + "pose_lift.txt",
                "f_intrinsics": general_output_folder + "intrinsics.txt"}

    if process_all_data:
        if not os.path.exists(openpose_liftnet_image_dir):
            os.makedirs(openpose_liftnet_image_dir)
        drone_flight_process_all_data(filenames, energy_parameters)

    if find_3d_pose:
        if not os.path.exists(gt_folder_dir):
            os.makedirs(gt_folder_dir)
        find_3d_gt(energy_parameters, filenames, mode)
