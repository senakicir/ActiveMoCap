from helpers import *
import torch
import numpy as np
from scipy.stats import pearsonr
import pdb

class Potential_Error_Finder(object):
    def __init__(self, general_param):

        self.find_best_traj = general_param["FIND_BEST_TRAJ"]
        self.predefined_traj_len = general_param["PREDEFINED_TRAJ_LEN"]
        self.num_of_noise_trials = general_param["NUM_OF_NOISE_TRIALS"]

        self.animation = general_param["ANIMATION_NUM"]

        self.frame_overall_error_list = np.zeros([self.num_of_noise_trials,])
        self.frame_current_error_list =  np.zeros([self.num_of_noise_trials,])
        self.frame_middle_error_list =  np.zeros([self.num_of_noise_trials,])

        self.correlation_current = []
        self.correlation_future = []
        self.cosine_current = []
        self.cosine_future = []

        self.overall_error_across_trials = [] 
        self.current_error_across_trials = [] 
        self.middle_error_across_trials = [] 

        self.overall_error_list = []
        self.current_error_list = []
        self.middle_error_list = []
        
        self.final_overall_error = 0
        self.final_current_error = 0
        self.final_middle_error = 0

    def append_error(self, trial_ind, adjusted_optimized_poses, poses_3d_gt, CURRENT_POSE_INDEX, MIDDLE_POSE_INDEX):
        self.frame_overall_error_list[trial_ind]  = np.mean(np.mean(np.linalg.norm(adjusted_optimized_poses[1:,:,:] - poses_3d_gt, axis=1), axis=1))
        self.frame_current_error_list[trial_ind]  = np.mean(np.linalg.norm(adjusted_optimized_poses[CURRENT_POSE_INDEX,:,:] - poses_3d_gt[CURRENT_POSE_INDEX-1,:,:], axis=0)) 
        self.frame_middle_error_list[trial_ind] =  np.mean(np.linalg.norm(adjusted_optimized_poses[MIDDLE_POSE_INDEX,:,:] - poses_3d_gt[MIDDLE_POSE_INDEX-1,:,:], axis=0)) 

    def record_noise_experiment_statistics(self, psf, state_ind):
        psf.overall_error_mean_list[state_ind],  psf.overall_error_std_list[state_ind] = np.mean(self.frame_overall_error_list), np.std(self.frame_overall_error_list)
        psf.current_error_mean_list[state_ind],  psf.current_error_std_list[state_ind] = np.mean(self.frame_current_error_list), np.std(self.frame_current_error_list)
        psf.middle_error_mean_list[state_ind],   psf.middle_error_std_list[state_ind]  = np.mean(self.frame_middle_error_list),  np.std(self.frame_middle_error_list)
        
        self.frame_overall_error_list = np.zeros([self.num_of_noise_trials,])
        self.frame_current_error_list =  np.zeros([self.num_of_noise_trials,])
        self.frame_middle_error_list =  np.zeros([self.num_of_noise_trials,])

        self.overall_error_across_trials.append(psf.overall_error_mean_list[state_ind])
        self.current_error_across_trials.append(psf.current_error_mean_list[state_ind])
        self.middle_error_across_trials.append(psf.middle_error_mean_list[state_ind])

    def find_correlations(self, psf):
        overall_uncertainty_arr = np.array(list(psf.uncertainty_list_whole.values()), dtype=float)
        norm_overall_uncertainty = (overall_uncertainty_arr-np.min(overall_uncertainty_arr))/(np.max(overall_uncertainty_arr)-np.min(overall_uncertainty_arr))
        norm_overall_error = (psf.overall_error_mean_list-np.min(psf.overall_error_mean_list))/(np.max(psf.overall_error_mean_list)-np.min(psf.overall_error_mean_list))
        self.correlation_current.append(pearsonr(norm_overall_uncertainty, norm_overall_error)[0])
        self.cosine_current.append(norm_overall_uncertainty@norm_overall_error/(np.linalg.norm(norm_overall_uncertainty)*np.linalg.norm(norm_overall_error)))

        #future_uncertainty_arr = np.array(list(psf.uncertainty_list_future.values()), dtype=float)
        #norm_future_uncertainty = (future_uncertainty_arr-np.min(future_uncertainty_arr))/(np.max(future_uncertainty_arr)-np.min(future_uncertainty_arr))
        #norm_future_error = (psf.future_error_mean_list-np.min(psf.future_error_mean_list))/(np.max(psf.future_error_mean_list)-np.min(psf.future_error_mean_list))
        #self.correlation_future.append(pearsonr(norm_future_uncertainty, norm_future_error)[0])
        #self.cosine_future.append(norm_future_uncertainty@norm_future_error/(np.linalg.norm(norm_future_uncertainty)*np.linalg.norm(norm_future_error)))

    def find_average_error_over_trials(self, index):
        overall_error_of_chosen_index = self.overall_error_across_trials[index]
        self.overall_error_list.append(overall_error_of_chosen_index)
        self.final_overall_error =  sum(self.overall_error_list)/len(self.overall_error_list)

        current_error_of_chosen_index = self.current_error_across_trials[index]
        self.current_error_list.append(current_error_of_chosen_index)
        self.final_current_error =  sum(self.current_error_list)/len(self.current_error_list)

        middle_error_of_chosen_index = self.middle_error_across_trials[index]
        self.middle_error_list.append(middle_error_of_chosen_index)
        self.final_middle_error =  sum(self.middle_error_list)/len(self.middle_error_list)

        self.overall_error_across_trials = [] 
        self.current_error_across_trials = [] 
        self.middle_error_across_trials = [] 


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

