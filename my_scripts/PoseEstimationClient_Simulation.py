from helpers import *
import pandas as pd
import torch
import numpy as np
from crop import Crop
from square_bounding_box import *
from kalman_filters import *
from project_bones import Projection_Client
from PoseEstimationClient import *
from scipy.stats import pearsonr
import pdb

class PoseEstimationClient_Simulation(PoseEstimationClient):
    def __init__(self, general_param):

        self.find_best_traj = general_param["FIND_BEST_TRAJ"]
        self.predefined_traj_len = general_param["PREDEFINED_TRAJ_LEN"]

        self.num_of_noise_trials = general_param["NUM_OF_NOISE_TRIALS"]
        self.animation = general_param["ANIMATION_NUM"]

        self.frame_overall_error_list = np.zeros([self.num_of_noise_trials,])
        self.frame_future_error_list =  np.zeros([self.num_of_noise_trials,])

        self.correlation_current = []
        self.correlation_future = []
        self.cosine_current = []
        self.cosine_future = []

        self.error_across_trials = [] 
        self.all_average_errors_across_trials = []
        self.final_average_error = 0
   
    def append_error(self, trial_ind, optimized_poses, poses_3d_gt):
        self.frame_overall_error_list[trial_ind]  = np.mean(np.mean(np.linalg.norm(optimized_poses[1:,:,:] - poses_3d_gt, axis=1), axis=1))
        self.frame_future_error_list[trial_ind]  = np.mean(np.linalg.norm(optimized_poses[0,:,:] - poses_3d_gt[0,:,:], axis=0)) 

    def record_noise_experiment_statistics(self, psf, state_ind):
        psf.overall_error_mean_list[state_ind], psf.future_error_mean_list[state_ind], psf.overall_error_std_list[state_ind], psf.future_error_std_list[state_ind] = np.mean(self.frame_overall_error_list), np.mean(self.frame_future_error_list), np.std(self.frame_overall_error_list), np.std(self.frame_future_error_list)
        self.error_across_trials.append(psf.overall_error_mean_list[state_ind])

    def find_correlations(self, psf):
        overall_uncertainty_arr = np.array(list(psf.uncertainty_list_whole.values()), dtype=float)
        norm_overall_uncertainty = (overall_uncertainty_arr-np.min(overall_uncertainty_arr))/(np.max(overall_uncertainty_arr)-np.min(overall_uncertainty_arr))
        norm_overall_error = (psf.overall_error_mean_list-np.min(psf.overall_error_mean_list))/(np.max(psf.overall_error_mean_list)-np.min(psf.overall_error_mean_list))
        self.correlation_current.append(pearsonr(norm_overall_uncertainty, norm_overall_error)[0])
        self.cosine_current.append(norm_overall_uncertainty@norm_overall_error/(np.linalg.norm(norm_overall_uncertainty)*np.linalg.norm(norm_overall_error)))

        future_uncertainty_arr = np.array(list(psf.uncertainty_list_future.values()), dtype=float)
        norm_future_uncertainty = (future_uncertainty_arr-np.min(future_uncertainty_arr))/(np.max(future_uncertainty_arr)-np.min(future_uncertainty_arr))
        norm_future_error = (psf.future_error_mean_list-np.min(psf.future_error_mean_list))/(np.max(psf.future_error_mean_list)-np.min(psf.future_error_mean_list))
        self.correlation_future.append(pearsonr(norm_future_uncertainty, norm_future_error)[0])
        self.cosine_future.append(norm_future_uncertainty@norm_future_error/(np.linalg.norm(norm_future_uncertainty)*np.linalg.norm(norm_future_error)))


    def find_average_error_over_trials(self, index):
        ave_error_of_chosen_index = self.error_across_trials[index]
        self.all_average_errors_across_trials.append(ave_error_of_chosen_index)
        self.final_average_error =  sum(self.all_average_errors_across_trials)/len(self.all_average_errors_across_trials)
        return ave_error_of_chosen_index

    def init_3d_pose(self, pose):
        if self.animation == "noise":
            self.prev_pose = pose.copy()

    def update_internal_3d_pose(self):
        if self.animation == "noise":
            self.prev_pose = add_noise_to_pose(torch.from_numpy(self.prev_pose).float(), self.pose_noise_3d_std).numpy()

    def adjust_3d_pose(self, current_state, pose_client_general):
        if self.animation == "noise":
            current_state.change_human_gt_info(self.prev_pose)
            self.update_bone_lengths(torch.from_numpy(self.prev_pose).float()) #this needs fixing~
            pose_client_general.update_bone_lengths(torch.from_numpy(self.prev_pose).float())

