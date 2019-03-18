import cv2 as cv2
from math import radians, cos, sin, pi, degrees, acos, sqrt
import numpy as np
import torch as torch
from helpers import range_angle, shape_cov, euler_to_rotation_matrix
import time as time 
from project_bones import take_potential_projection, CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET
import pdb 

#constants
BETA = 0.35

INCREMENT_DEGREE_AMOUNT = radians(-20)
INCREMENT_RADIUS = 3

z_pos = -1.5
DELTA_T = 0.2
N = 4.0
TIME_HORIZON = N*DELTA_T
SAFE_RADIUS = 7
TOP_SPEED = 3

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
    def __init__(self, use_single_joint, model_settings):
        self.bone_connections, self.joint_names, self.num_of_joints, self.hip_index = model_settings
        self.human_pos_est = np.array([0,0,0])
        self.human_pos_gt = np.array([0,0,0])
        self.left_arm_ind = self.joint_names.index('left_arm')
        self.right_arm_ind = self.joint_names.index('right_arm')
        if use_single_joint:
            self.left_arm_ind = 0
            self.right_arm_ind = 0

        self.radius = SAFE_RADIUS#np.linalg.norm(projected_distance_vect[0:2,]) #to do

        #drone_polar_pos = np.array([0,0,0])#positions_[HUMAN_POS_IND, :] #find the drone initial angle (needed for trackbar)
        #self.some_angle = range_angle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)

        self.R_drone_gt = torch.zeros([3,3])
        self.C_drone_gt = torch.zeros([3,1])
        self.R_cam_gt = torch.zeros([3,3])

        self.human_orientation_gt = np.zeros([3,])
        self.drone_orientation_gt = np.zeros([3,])
        self.drone_pos_gt = np.zeros([3,])
        self.human_pos_gt = np.zeros([3,])
        self.bone_pos_gt = np.zeros([3, self.num_of_joints])

        self.drone_pos_est = np.zeros([3,])
        self.human_pos_est = np.zeros([3,])
        self.human_orientation_est = np.zeros([3,])
        self.drone_orientation_est = np.array([0,0,0])
        self.bone_pos_est = np.zeros([3, self.num_of_joints])
        self.cam_pitch = 0

    def change_human_gt_info(self, bone_pos_gt_updated):
        self.bone_pos_gt =  bone_pos_gt_updated.copy()
        self.human_pos_gt = self.bone_pos_gt[:, self.hip_index]

        shoulder_vector_gt = self.bone_pos_gt[:, self.left_arm_ind] - self.bone_pos_gt[:, self.right_arm_ind] 
        self.human_orientation_gt = np.arctan2(-shoulder_vector_gt[0], shoulder_vector_gt[1])

    def store_frame_parameters(self, bone_pos_gt, drone_orientation_gt, drone_pos_gt, drone_pos_est):
        self.change_human_gt_info(bone_pos_gt)
        
        self.drone_orientation_gt = drone_orientation_gt
        self.drone_pos_gt = drone_pos_gt

        self.R_drone_gt = euler_to_rotation_matrix(self.drone_orientation_gt[0], self.drone_orientation_gt[1], self.drone_orientation_gt[2])
        self.C_drone_gt = torch.from_numpy(drone_pos_gt).float()
        self.R_cam_gt = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, self.cam_pitch+pi/2, CAMERA_YAW_OFFSET)

        self.drone_pos_est = drone_pos_est

    def get_frame_parameters(self):
        return self.bone_pos_gt, self.R_drone_gt, self.C_drone_gt, self.R_cam_gt

    def update_human_info(self, bone_pos_est):
        self.bone_pos_est = bone_pos_est
        shoulder_vector_gt = bone_pos_est[:, self.left_arm_ind] - bone_pos_est[:, self.right_arm_ind] 
        self.human_orientation_est = np.arctan2(-shoulder_vector_gt[0], shoulder_vector_gt[1])
        self.human_pos_est = bone_pos_est[:, self.hip_index]

    def get_goal_pos_yaw_pitch(self, goal_state):
        goal_pos = goal_state["position"]
        goal_yaw = goal_state["orientation"]
        cam_pitch = goal_state["pitch"]
        desired_yaw_deg = find_delta_yaw((self.drone_orientation_gt)[2],  goal_yaw)
        return goal_pos , desired_yaw_deg, cam_pitch   

    def get_required_pitch(self):
        new_radius = np.linalg.norm(self.drone_pos_gt - self.human_pos_est)
        new_theta = acos((self.drone_pos_gt[2] - self.human_pos_est[2])/new_radius)
        new_pitch = pi/2 - new_theta
        return new_pitch

        ############ USELESS

    def get_delta_orient(self, target_yaw):
        delta_yaw = find_delta_yaw((self.drone_orientation)[2],  target_yaw)
        return delta_yaw

    def get_desired_pos_and_yaw_trackbar(self):
        #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
        input_rad = radians(cv2.getTrackbarPos('Angle', 'Drone Control')) #according to what degree we want the drone to be at
        current_radius = cv2.getTrackbarPos('Radius', 'Drone Control')
        desired_z_pos = cv2.getTrackbarPos('Z', 'Drone Control')
        #input_rad_unreal_orient = input_rad + INITIAL_HUMAN_ORIENTATION #we don't use this at all currently
        #desired_polar_angle = state.human_orientation + input_rad + state.human_rotation_speed*TIME_HORIZON
        desired_polar_angle = input_rad

        desired_polar_pos = np.array([cos(desired_polar_angle) * current_radius, sin(desired_polar_angle) * current_radius, 0])
        #desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel - np.array([0,0,desired_z_pos])
        desired_pos = desired_polar_pos + self.human_pos - np.array([0,0,desired_z_pos])
        desired_yaw = desired_polar_angle - pi
        return desired_pos, desired_yaw

    def get_desired_pos_and_angle_fixed_rotation(self):
        desired_polar_angle = self.current_degree + INCREMENT_DEGREE_AMOUNT
        desired_polar_pos = np.array([cos(desired_polar_angle) * self.radius, sin(desired_polar_angle) * self.radius, 0])
        desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel 
        desired_pos[2] = self.human_pos[2]-z_pos
        desired_yaw = self.current_degree + pi#INCREMENT_DEGREE_AMOUNT/N + pi
        desired_yaw_deg = find_delta_yaw((self.drone_orientation)[2], desired_yaw)

        return desired_pos, desired_yaw_deg