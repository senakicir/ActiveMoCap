from helpers import add_noise_to_pose
import torch
import numpy as np
from State import State


class Dome_Experiment_Client(object):
    def __init__(self, param):
        self.find_best_traj = param["FIND_BEST_TRAJ"]
        self.predefined_traj_len = param["PREDEFINED_TRAJ_LEN"]
        self.anim_num = param["ANIMATION_NUM"]
        self.prev_pose = 0
        self.pose_noise_3d_std = param["POSE_NOISE_3D_STD"]
        self.num_of_noise_trials = param["NUM_OF_NOISE_TRIALS"]

        self.frame_overall_error_list = np.zeros([self.num_of_noise_trials,])
        self.frame_future_error_list =  np.zeros([self.num_of_noise_trials,])

    def init_3d_pose(self, pose):
        self.prev_pose = pose.copy()

    def adjust_3d_pose(self, current_state, pose_client):
        if self.anim_num == "noise":
            self.prev_pose = add_noise_to_pose(self.prev_pose, self.pose_noise_3d_std)
            current_state.change_human_gt_info(self.prev_pose)
            pose_client.update_bone_lengths(self.prev_pose)
        else:
            self.prev_pose = current_state.bone_pos_gt

    def record_noise_experiment_statistics(self, psf, state_ind):
        psf.overall_error_list[state_ind], psf.future_error_list[state_ind], psf.overall_error_std_list[state_ind], psf.future_error_std_list[state_ind] = np.mean(self.frame_overall_error_list), np.mean(self.frame_future_error_list), np.std(self.frame_overall_error_list), np.std(self.frame_future_error_list)