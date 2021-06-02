import cv2 as cv2
from math import radians, cos, sin, pi, degrees, acos, sqrt
import numpy as np
import torch as torch
from helpers import range_angle, shape_cov, euler_to_rotation_matrix
import time as time 
from pose_helper_functions import add_noise_to_pose
from Projection_Client import Projection_Client, CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET, neat_tensor

#constants
BETA = 0.35

INCREMENT_DEGREE_AMOUNT = radians(-20)
INCREMENT_RADIUS = 3
SAVED_VALS_INITIAL_DRONE_POS =  np.array([[-61640.0,-23177.0,-19693.0]]).T


z_pos = -1.5
N = 4.0

def find_current_polar_info(drone_pos, human_pos):
    polar_pos = drone_pos - human_pos  #subtrack the human_pos in order to find the current polar position vector.
    polar_degree = np.arctan2(polar_pos[1], polar_pos[0])  #NOT relative to initial human angle, not using currently
    return polar_pos, polar_degree

def find_delta_yaw(current_yaw, desired_yaw):
    current_yaw_deg = degrees(current_yaw)
    yaw_candidates = np.array([degrees(desired_yaw), degrees(desired_yaw) - 360, degrees(desired_yaw) +360])
    min_diff = np.array([abs(current_yaw_deg -  yaw_candidates[0]), abs(current_yaw_deg -  yaw_candidates[1]), abs(current_yaw_deg -  yaw_candidates[2])])
    return yaw_candidates[np.argmin(min_diff)]

def find_pose_at_time (anim_time, search_array, num_of_joints):
    flat_pose = search_array[abs(search_array[:,0]-anim_time)<1e-4, 1:]
    fail_msg = "trying to search for anim at time {}".format(str(anim_time))
    assert flat_pose.size != 0, anim_time
    pose = flat_pose.reshape([3,num_of_joints], order="F")
    return pose, None

def find_pose_and_frame_at_time (anim_time, search_array, num_of_joints):
    flat_pose = search_array[abs(search_array[:,1]-anim_time)<1e-4, 2:]
    fail_msg = "trying to search for anim at time {}".format(str(anim_time))
    assert flat_pose.size != 0, fail_msg
    pose = flat_pose.reshape([3,num_of_joints], order="F")
    frame_num = search_array[abs(search_array[:,1]-anim_time)<1e-4, 0]
    return pose, int(frame_num[0])

def find_human_pose_orientation(pose_3d, left_arm_ind, right_arm_ind):
    shoulder_vector_gt = pose_3d[:, left_arm_ind] - pose_3d[:, right_arm_ind] 
    return np.arctan2(-shoulder_vector_gt[0], shoulder_vector_gt[1])

def convert_saved_vals(saved_pose_gt, initial_drone_pos):
    return ((saved_pose_gt*100+SAVED_VALS_INITIAL_DRONE_POS)-initial_drone_pos[:, np.newaxis])/100

class State(object):
    def __init__(self, use_single_joint, active_parameters, model_settings, anim_gt_array, future_window_size,  initial_drone_pos, camera_offset_x):
        self.model_settings = model_settings
        self.bone_connections, self.joint_names, self.num_of_joints, self.hip_index = self.model_settings
        self.anim_gt_array = anim_gt_array
        self.active_parameters = active_parameters 
        self.future_window_size = future_window_size
        self.initial_drone_pos = initial_drone_pos
        self.camera_offset_x = camera_offset_x
        self.C_cam_torch = torch.FloatTensor([[self.camera_offset_x], [0], [0]])

        self.left_arm_ind = self.joint_names.index('left_arm')
        self.right_arm_ind = self.joint_names.index('right_arm')
        self.use_single_joint = use_single_joint
        if use_single_joint:
            self.left_arm_ind = 0
            self.right_arm_ind = 0

        self.radius = active_parameters["SAFE_RADIUS"]#np.linalg.norm(projected_distance_vect[0:2,]) #to do
        self.TOP_SPEED = active_parameters["TOP_SPEED"]
        self.DELTA_T = active_parameters["DELTA_T"]

        #drone_polar_pos = np.array([0,0,0])#positions_[HUMAN_POS_IND, :] #find the drone initial angle (needed for trackbar)
        #self.some_angle = range_angle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)

        self.R_drone_gt = torch.zeros([3,3])
        self.C_drone_gt = torch.zeros([3,1])
        self.prev_C_drone_gt = torch.zeros([3,1])
        self.R_cam_gt = torch.zeros([3,3])

        self.human_orientation_gt = 0
        self.drone_orientation_gt = np.zeros([3,])
        self.human_pos_gt = np.zeros([3,])
        self.bone_pos_gt = np.zeros([3, self.num_of_joints])
        self.current_drone_vel = np.zeros([3,])

        self.drone_transformation_matrix = torch.zeros(4,4)
        self.inv_drone_transformation_matrix = torch.zeros(4,4)

        self.human_pos_est = np.zeros([3,])
        self.human_orientation_est = 0
        self.drone_orientation_est = np.zeros([3,])
        self.drone_pos_est = np.zeros([3,1])
        self.bone_pos_est = np.zeros([3, self.num_of_joints])
        self.__cam_pitch__ = 0
        self.camera_id = 0

        self.anim_time = 1
        self.futuremost_pose_3d_gt = None
        self.function_find_pose = None

        self.distances_travelled = []
        self.total_distance_travelled = 0

    def init_anim_time(self, anim_time, animation):
        self.anim_time = anim_time

        future_anim_time = self.anim_time + self.DELTA_T*self.future_window_size
        if animation == "mpi_inf_3dhp" or animation == "cmu_panoptic_pose_1" or animation == "cmu_panoptic_dance_3":
            self.function_find_pose = find_pose_and_frame_at_time
        else:
            self.function_find_pose = find_pose_at_time

        if self.anim_gt_array is not None:
            saved_pose_gt, _ = self.function_find_pose(future_anim_time, self.anim_gt_array, self.num_of_joints)
            self.futuremost_pose_3d_gt = convert_saved_vals(saved_pose_gt, self.initial_drone_pos)

    def update_animation_gt_array(self, new_anim_gt_array):
        self.anim_gt_array = new_anim_gt_array

    def set_cam_pitch(self, new_pitch):
        self.__cam_pitch__ = new_pitch

    def get_cam_pitch(self):
        return self.__cam_pitch__

    def deepcopy_state(self):
        new_state = State(self.use_single_joint, self.active_parameters, self.model_settings, self.anim_gt_array, self.future_window_size, self.initial_drone_pos, self.camera_offset_x)

        new_state.R_drone_gt = self.R_drone_gt.clone()
        new_state.C_drone_gt = self.C_drone_gt.clone()
        new_state.prev_C_drone_gt = self.prev_C_drone_gt.clone()
        new_state.R_cam_gt = self.R_cam_gt.clone()

        new_state.human_orientation_gt = self.human_orientation_gt        
        new_state.drone_orientation_gt = self.drone_orientation_gt.copy()
        new_state.human_pos_gt = self.human_pos_gt.copy()
        new_state.bone_pos_gt = self.bone_pos_gt.copy()
        if self.current_drone_vel is not None:
            new_state.current_drone_vel = self.current_drone_vel.copy()

        new_state.drone_transformation_matrix = self.drone_transformation_matrix.clone()
        new_state.inv_drone_transformation_matrix = self.inv_drone_transformation_matrix.clone()

        new_state.human_pos_est = self.human_pos_est.copy()
        new_state.human_orientation_est = self.human_orientation_est
        new_state.drone_orientation_est = self.drone_orientation_est.copy()
        new_state.drone_pos_est = self.drone_pos_est.copy()
        new_state.bone_pos_est = self.bone_pos_est.copy()

        new_state.set_cam_pitch(self.__cam_pitch__)
        new_state.anim_time = self.anim_time
        new_state.camera_id = self.camera_id
        
        if self.anim_gt_array is not None:
            new_state.futuremost_pose_3d_gt = self.futuremost_pose_3d_gt.copy()

        new_state.function_find_pose = self.function_find_pose
        new_state.total_distance_travelled = self.total_distance_travelled
        new_state.distances_travelled = self.distances_travelled.copy()
    
        return new_state

    def change_human_gt_info(self, bone_pos_gt_updated):
        self.bone_pos_gt =  bone_pos_gt_updated.copy()
        self.human_pos_gt = self.bone_pos_gt[:, self.hip_index]
        self.human_orientation_gt = find_human_pose_orientation(self.bone_pos_gt, self.left_arm_ind, self.right_arm_ind)

    def compare_arrays(self, bone_pos_gt):
        if self.anim_gt_array is not None:
            saved_pose_gt, _ = self.function_find_pose(self.anim_time, self.anim_gt_array, self.num_of_joints)
            converted_saved_pose_gt = convert_saved_vals(saved_pose_gt, self.initial_drone_pos)
            fail_msg = "The distances are: %{}".format(np.linalg.norm(bone_pos_gt-converted_saved_pose_gt, axis=1))
            assert np.allclose(bone_pos_gt,converted_saved_pose_gt, atol=1e-03), fail_msg

    def store_frame_parameters(self, bone_pos_gt, drone_orientation_gt, drone_pos_gt, drone_vel, camera_id):
        self.camera_id = camera_id
        self.change_human_gt_info(bone_pos_gt)
        
        self.drone_orientation_gt = drone_orientation_gt
        self.current_drone_vel = drone_vel

        self.R_drone_gt = euler_to_rotation_matrix(self.drone_orientation_gt[0], self.drone_orientation_gt[1], self.drone_orientation_gt[2])
        self.C_drone_gt = torch.from_numpy(drone_pos_gt).float()
        self.R_cam_gt = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, self.__cam_pitch__+pi/2, CAMERA_YAW_OFFSET)

        #form drone translation matrix (similar to dataset)
        drone_transformation = torch.cat((torch.cat((self.R_drone_gt, self.C_drone_gt), dim=1), neat_tensor), dim=0)
        camera_transformation = torch.cat((torch.cat((self.R_cam_gt, self.C_cam_torch), dim=1), neat_tensor), dim=0) 
        self.drone_transformation_matrix = drone_transformation@camera_transformation
        self.inv_drone_transformation_matrix = torch.inverse(self.drone_transformation_matrix)

        if not torch.allclose(self.prev_C_drone_gt, torch.zeros(1)):
            self.distances_travelled.append(torch.norm(self.prev_C_drone_gt[:,0] -self.C_drone_gt[:,0]))
            self.total_distance_travelled = sum(self.distances_travelled)
        self.prev_C_drone_gt = self.C_drone_gt.clone()


    def store_frame_transformation_matrix_joint_gt(self, bone_pos_gt, drone_transformation_matrix, camera_id):
        self.camera_id = camera_id
        self.change_human_gt_info(bone_pos_gt)

        self.R_cam_gt = torch.zeros([3,3])

        self.drone_orientation_gt = np.zeros([3,])

        self.drone_transformation_matrix = drone_transformation_matrix.clone()
        self.R_drone_gt = self.drone_transformation_matrix[0:3,0:3]
        self.C_drone_gt = self.drone_transformation_matrix[0:3,3].unsqueeze(1)

        self.inv_drone_transformation_matrix = torch.inverse(self.drone_transformation_matrix)

    def get_frame_parameters(self):
        return self.camera_id, self.bone_pos_gt.copy(), self.futuremost_pose_3d_gt, self.inv_drone_transformation_matrix.clone(), self.drone_transformation_matrix.clone()

    def update_human_info(self, bone_pos_est):
        self.bone_pos_est = bone_pos_est.copy()
        shoulder_vector_gt = bone_pos_est[:, self.left_arm_ind] - bone_pos_est[:, self.right_arm_ind] 
        self.human_orientation_est = np.arctan2(-shoulder_vector_gt[0], shoulder_vector_gt[1])
        self.human_pos_est = bone_pos_est[:, self.hip_index].copy()

    def update_anim_time(self, anim_time):
        self.anim_time = anim_time
        future_anim_time = self.anim_time + self.DELTA_T*self.future_window_size
        if self.anim_gt_array is not None:
            saved_pose_gt, _ = self.function_find_pose(future_anim_time, self.anim_gt_array, self.num_of_joints)
            converted_saved_pose_gt = convert_saved_vals(saved_pose_gt, self.initial_drone_pos)
            self.futuremost_pose_3d_gt = converted_saved_pose_gt

    def get_first_future_poses(self):
        future_poses_3d_gt = np.zeros([self.future_window_size, 3, self.num_of_joints])
        for future_ind in range(self.future_window_size):
            future_anim_time = self.anim_time + self.DELTA_T*(future_ind+1)
            saved_pose_gt, _ = self.function_find_pose(future_anim_time, self.anim_gt_array, self.num_of_joints)
            converted_saved_pose_gt = convert_saved_vals(saved_pose_gt, self.initial_drone_pos)
            future_poses_3d_gt[self.future_window_size-future_ind-1,:,:] = converted_saved_pose_gt.copy()
        return future_poses_3d_gt

    def get_required_pitch(self):
        new_radius = np.linalg.norm(self.C_drone_gt.numpy() - self.human_pos_est)
        new_theta = acos((self.C_drone_gt[2] - self.human_pos_est[2])/new_radius)
        new_pitch = pi/2 - new_theta
        return new_pitch