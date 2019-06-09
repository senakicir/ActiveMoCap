import cv2 as cv2
from math import radians, cos, sin, pi, degrees, acos, sqrt
import numpy as np
import torch as torch
from helpers import range_angle, shape_cov, euler_to_rotation_matrix
import time as time 
from pose_helper_functions import add_noise_to_pose
from project_bones import Projection_Client, CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET, neat_tensor, C_cam_torch
import pdb 

#constants
BETA = 0.35

INCREMENT_DEGREE_AMOUNT = radians(-20)
INCREMENT_RADIUS = 3

z_pos = -1.5
N = 4.0
SAFE_RADIUS = 7

def find_current_polar_info(drone_pos, human_pos):
    polar_pos = drone_pos - human_pos  #subtrack the human_pos in order to find the current polar position vector.
    polar_degree = np.arctan2(polar_pos[1], polar_pos[0])  #NOT relative to initial human angle, not using currently
    return polar_pos, polar_degree

def find_delta_yaw(current_yaw, desired_yaw):
    current_yaw_deg = degrees(current_yaw)
    yaw_candidates = np.array([degrees(desired_yaw), degrees(desired_yaw) - 360, degrees(desired_yaw) +360])
    min_diff = np.array([abs(current_yaw_deg -  yaw_candidates[0]), abs(current_yaw_deg -  yaw_candidates[1]), abs(current_yaw_deg -  yaw_candidates[2])])
    return yaw_candidates[np.argmin(min_diff)]


class State(object):
    def __init__(self, use_single_joint, active_parameters, model_settings):
        self.bone_connections, self.joint_names, self.num_of_joints, self.hip_index = model_settings
        self.active_parameters = active_parameters 

        self.left_arm_ind = self.joint_names.index('left_arm')
        self.right_arm_ind = self.joint_names.index('right_arm')
        self.use_single_joint = use_single_joint
        if use_single_joint:
            self.left_arm_ind = 0
            self.right_arm_ind = 0

        self.radius = SAFE_RADIUS#np.linalg.norm(projected_distance_vect[0:2,]) #to do
        
        self.TOP_SPEED = active_parameters["TOP_SPEED"]
        self.DELTA_T = active_parameters["DELTA_T"]

        #drone_polar_pos = np.array([0,0,0])#positions_[HUMAN_POS_IND, :] #find the drone initial angle (needed for trackbar)
        #self.some_angle = range_angle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)

        self.R_drone_gt = torch.zeros([3,3])
        self.C_drone_gt = torch.zeros([3,1])
        self.R_cam_gt = torch.zeros([3,3])

        self.human_orientation_gt = np.zeros([3,])
        self.drone_orientation_gt = np.zeros([3,])
        self.human_pos_gt = np.zeros([3,])
        self.bone_pos_gt = np.zeros([3, self.num_of_joints])

        self.drone_transformation_matrix = torch.zeros(4,4)
        self.inv_drone_transformation_matrix = torch.zeros(4,4)

        self.human_pos_est = np.zeros([3,])
        self.human_orientation_est = np.zeros([3,])
        self.drone_orientation_est = np.zeros([3,])
        self.drone_pos_est = np.zeros([3,1])
        self.bone_pos_est = np.zeros([3, self.num_of_joints])
        self.cam_pitch = 0

    def deepcopy_state(self):
        model_settings =[self.bone_connections, self.joint_names, self.num_of_joints, self.hip_index]
        new_state = State(self.use_single_joint, self.active_parameters,  model_settings)

        new_state.R_drone_gt = self.R_drone_gt.clone()
        new_state.C_drone_gt = self.C_drone_gt.clone()
        new_state.R_cam_gt = self.R_cam_gt.clone()

        new_state.human_orientation_gt = self.human_orientation_gt.copy()
        new_state.drone_orientation_gt = self.drone_orientation_gt.copy()
        new_state.human_pos_gt = self.human_pos_gt.copy()
        new_state.bone_pos_gt = self.bone_pos_gt.copy()

        new_state.drone_transformation_matrix = self.drone_transformation_matrix.clone()
        new_state.inv_drone_transformation_matrix = self.inv_drone_transformation_matrix.clone()

        new_state.human_pos_est = self.human_pos_est.copy()
        new_state.human_orientation_est = self.human_orientation_est.copy()
        new_state.drone_orientation_est = self.drone_orientation_est.copy()
        new_state.drone_pos_est = self.drone_pos_est.copy()
        new_state.bone_pos_est = self.bone_pos_est.copy()
        new_state.cam_pitch = self.cam_pitch
        return new_state

    def change_human_gt_info(self, bone_pos_gt_updated):
        self.bone_pos_gt =  bone_pos_gt_updated.copy()
        self.human_pos_gt = self.bone_pos_gt[:, self.hip_index]

        shoulder_vector_gt = self.bone_pos_gt[:, self.left_arm_ind] - self.bone_pos_gt[:, self.right_arm_ind] 
        self.human_orientation_gt = np.arctan2(-shoulder_vector_gt[0], shoulder_vector_gt[1])

    def store_frame_parameters(self, bone_pos_gt, drone_orientation_gt, drone_pos_gt, drone_pos_est):
        self.change_human_gt_info(bone_pos_gt)
        
        self.drone_orientation_gt = drone_orientation_gt

        self.R_drone_gt = euler_to_rotation_matrix(self.drone_orientation_gt[0], self.drone_orientation_gt[1], self.drone_orientation_gt[2])
        self.C_drone_gt = torch.from_numpy(drone_pos_gt).float()
        self.R_cam_gt = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, self.cam_pitch+pi/2, CAMERA_YAW_OFFSET)

        self.drone_pos_est = drone_pos_est

        #form drone translation matrix (similar to dataset)
        drone_transformation = torch.cat((torch.cat((self.R_drone_gt, self.C_drone_gt), dim=1), neat_tensor), dim=0)
        camera_transformation = torch.cat((torch.cat((self.R_cam_gt, C_cam_torch), dim=1), neat_tensor), dim=0) 
        self.drone_transformation_matrix = drone_transformation@camera_transformation
        self.inv_drone_transformation_matrix = torch.inverse(self.drone_transformation_matrix)

    def store_frame_transformation_matrix_joint_gt(self, bone_pos_gt, drone_transformation_matrix):
        self.change_human_gt_info(bone_pos_gt)

        self.R_cam_gt = torch.zeros([3,3])

        self.drone_orientation_gt = np.zeros([3,])

        self.drone_transformation_matrix = drone_transformation_matrix.clone()
        self.R_drone_gt = self.drone_transformation_matrix[0:3,0:3]
        self.C_drone_gt = self.drone_transformation_matrix[0:3,3].unsqueeze(1)

        self.inv_drone_transformation_matrix = torch.inverse(self.drone_transformation_matrix)

    def get_frame_parameters(self):
        return self.bone_pos_gt.copy(), self.inv_drone_transformation_matrix.clone(), self.drone_transformation_matrix.clone()

    def update_human_info(self, bone_pos_est):
        self.bone_pos_est = bone_pos_est.copy()
        shoulder_vector_gt = bone_pos_est[:, self.left_arm_ind] - bone_pos_est[:, self.right_arm_ind] 
        self.human_orientation_est = np.arctan2(-shoulder_vector_gt[0], shoulder_vector_gt[1])
        self.human_pos_est = bone_pos_est[:, self.hip_index].copy()

    def get_required_pitch(self):
        new_radius = np.linalg.norm(self.C_drone_gt.numpy() - self.human_pos_est)
        new_theta = acos((self.C_drone_gt[2] - self.human_pos_est[2])/new_radius)
        new_pitch = pi/2 - new_theta
        return new_pitch