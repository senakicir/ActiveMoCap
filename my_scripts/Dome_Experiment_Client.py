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

    def init_3d_pose(self, pose):
        self.prev_pose = pose.copy()

    def adjust_3d_pose(self, current_state, pose_client):
        if self.anim_num == "noise":
            self.prev_pose = add_noise_to_pose(self.prev_pose, self.pose_noise_3d_std)
            current_state.change_human_gt_info(self.prev_pose)
        else:
            self.prev_pose = current_state.bone_pos_gt