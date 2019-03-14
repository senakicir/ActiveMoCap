from helpers import *
import pandas as pd
import torch
import numpy as np
from crop import Crop
from square_bounding_box import *
from kalman_filters import *
from project_bones import take_bone_backprojection_pytorch
from PoseEstimationClient import *

class PoseEstimationClient_Simulation(PoseEstimationClient):
    def __init__(self, param, cropping_tool, pose_client_general):
        PoseEstimationClient.__init__(self, param, cropping_tool)
        self.future_proj_mode = True
        self.update_initial_param(pose_client_general)
        self.rewind_step()        
        self.modes["mode_2d"]=0        
    
    def update_initial_param(self, pose_client_general):
        self.init_optimized_poses = pose_client_general.optimized_poses.copy()
        self.init_pose_3d_preoptimization = pose_client_general.pose_3d_preoptimization.copy() 
        self.init_requiredEstimationData = pose_client_general.requiredEstimationData.copy()
        self.init_liftPoseList = pose_client_general.liftPoseList.copy()
        self.init_poses_3d_gt = pose_client_general.poses_3d_gt.copy()
        self.init_middle_pose_error = pose_client_general.middle_pose_error.copy()

        self.init_calib_res_list = pose_client_general.calib_res_list.copy()
        self.init_online_res_list = pose_client_general.online_res_list.copy()
        self.init_processing_time = pose_client_general.processing_time.copy()
        self.init_middle_pose_GT_list = pose_client_general.middle_pose_GT_list.copy()

        self.init_error_3d = pose_client_general.error_3d.copy()
        self.init_error_2d = pose_client_general.error_2d.copy()
        self.isCalibratingEnergy = pose_client_general.isCalibratingEnergy
        self.result_shape, self.result_size, self.loss_dict = pose_client_general.result_shape, pose_client_general.result_size, pose_client_general.loss_dict

    def rewind_step(self):
        self.optimized_poses = self.init_optimized_poses.copy()
        self.pose_3d_preoptimization = self.init_pose_3d_preoptimization.copy()
        self.requiredEstimationData = self.init_requiredEstimationData.copy()
        self.liftPoseList = self.init_liftPoseList.copy()
        self.poses_3d_gt = self.init_poses_3d_gt.copy()

        self.middle_pose_error = self.init_middle_pose_error.copy()
        self.error_3d = self.init_error_3d.copy()

        self.calib_res_list = self.init_calib_res_list.copy()
        self.online_res_list = self.init_online_res_list.copy()
        self.processing_time = self.init_processing_time.copy()

        self.error_3d = self.init_error_3d.copy()
        self.error_2d = self.init_error_2d.copy()


###edit
    def addNewFrame(self, pose_2d, R_drone, C_drone, R_cam, linecount, pose_3d_gt, pose3d_lift):
        self.liftPoseList.insert(0, pose3d_lift)
        self.requiredEstimationData.insert(0, [pose_2d, R_drone, C_drone, R_cam])

        temp = self.poses_3d_gt[:-1,:].copy() 
        self.poses_3d_gt[0,:] = pose_3d_gt.copy()
        self.poses_3d_gt[1:,:] = temp.copy()
        
    def update3dPos(self, optimized_poses):
        self.optimized_poses = optimized_poses.copy()
        
    def update_middle_pose_GT(self, middle_pose):
        pass

    def initialize_pose_3d(self, pose_3d_gt, calculating_future, linecount, pose_2d, R_drone_gt, C_drone_gt, R_cam_gt):
        self.pose_3d_preoptimization = self.optimized_poses.copy()

    def get_error(self):
        overall_error = np.mean(np.linalg.norm(self.optimized_poses - self.poses_3d_gt, axis=1))
        current_error = np.mean(np.linalg.norm(self.optimized_poses[0,:,:] - self.poses_3d_gt[0,:,:], axis=0))
        return overall_error, current_error
