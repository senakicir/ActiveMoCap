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

        self.update_initial_param(pose_client_general)
        self.rewind_step()        
    
    def update_initial_param(self, pose_client_general):
        self.init_optimized_poses = pose_client_general.optimized_poses.copy()
        self.init_pose_3d_preoptimization = pose_client_general.pose_3d_preoptimization.copy() 
        self.init_requiredEstimationData = pose_client_general.requiredEstimationData.copy()
        self.init_liftPoseList = pose_client_general.liftPoseList.copy()
        self.init_pose_list_3d_gt = pose_client_general.pose_list_3d_gt.copy()
        self.init_middle_pose_error = pose_client_general.middle_pose_error.copy()

        self.init_calib_res_list = pose_client_general.calib_res_list.copy()
        self.init_online_res_list = pose_client_general.online_res_list.copy()
        self.init_processing_time = pose_client_general.processing_time.copy()
        self.init_middle_pose_GT_list = pose_client_general.middle_pose_GT_list.copy()

        self.init_error_3d = pose_client_general.error_3d.copy()
        self.init_error_2d = pose_client_general.error_2d.copy()

    def rewind_step(self):
        self.optimized_poses = self.init_optimized_poses.copy()
        self.pose_3d_preoptimization = self.init_pose_3d_preoptimization.copy()
        self.requiredEstimationData = self.init_requiredEstimationData.copy()
        self.liftPoseList = self.init_liftPoseList.copy()
        self.pose_list_3d_gt = self.init_pose_list_3d_gt.copy()

        self.middle_pose_error = self.init_middle_pose_error.copy()
        self.error_3d = self.init_error_3d.copy()

        self.calib_res_list = self.init_calib_res_list.copy()
        self.online_res_list = self.init_online_res_list.copy()
        self.processing_time = self.init_processing_time.copy()

        self.error_3d = self.init_error_3d.copy()
        self.error_2d = self.init_error_2d.copy()


###edit
    def addNewFrame(self, pose_2d, R_drone, C_drone, R_cam, linecount, pose_3d_gt, pose3d_lift = None):
        self.threw_Away = False
        self.liftPoseList.insert(0, pose3d_lift)
        self.requiredEstimationData.insert(0, [pose_2d, R_drone, C_drone, R_cam])
        self.pose_list_3d_gt.insert(0, pose_3d_gt)
        
        if self.isCalibratingEnergy:
            if linecount >= self.PRECALIBRATION_LENGTH:
                while len(self.requiredEstimationData) > self.CALIBRATION_WINDOW_SIZE:
                    self.thrownAway_requiredEstimationData = self.requiredEstimationData.pop()
                    self.thrownAway_liftPoseList = self.liftPoseList.pop()
                    self.threw_Away = True

        else:
            if (len(self.requiredEstimationData) > self.ONLINE_WINDOW_SIZE):
                self.thrownAway_requiredEstimationData = self.requiredEstimationData.pop()
                self.thrownAway_liftPoseList = self.liftPoseList.pop()
                self.thrownAway_3d_gt = self.pose_list_3d_gt.pop()
                self.threw_Away = True

    def update3dPos(self, optimized_poses):
        self.prev_optimized_poses = self.optimized_poses.copy()

        if (self.isCalibratingEnergy):
            self.current_pose = optimized_poses.copy()
            self.middle_pose = optimized_poses.copy()
            self.future_pose = optimized_poses.copy()
            self.optimized_poses = np.repeat(optimized_poses[np.newaxis, :, :], self.ONLINE_WINDOW_SIZE, axis=0).copy()

        else:
            self.current_pose =  optimized_poses[CURRENT_POSE_INDEX, :,:].copy() #current pose
            self.middle_pose = optimized_poses[MIDDLE_POSE_INDEX, :,:].copy() #middle_pose
            self.future_pose =  optimized_poses[FUTURE_POSE_INDEX, :,:].copy() #future pose
            self.optimized_poses = optimized_poses[1:,:,:].copy()
        
    def update_middle_pose_GT(self, middle_pose):
        self.middle_pose_GT_list.insert(0, middle_pose)
        if (len(self.middle_pose_GT_list) > MIDDLE_POSE_INDEX):
            self.middle_pose_GT_list.pop()
        return self.middle_pose_GT_list[-1]

    def initialize_pose_3d(self, pose_3d_gt, calculating_future, linecount, pose_2d, R_drone_gt, C_drone_gt, R_cam_gt):
        if (linecount != 0):
            current_frame_init = self.future_pose.copy()
        else:
            if self.init_pose_with_gt:
                current_frame_init = pose_3d_gt.copy()
            else:
                current_frame_init = take_bone_backprojection_pytorch(pose_2d, R_drone_gt, C_drone_gt, R_cam_gt, self.joint_names).numpy()
        
        if self.isCalibratingEnergy:
            self.pose_3d_preoptimization = current_frame_init.copy()
        else:
            self.pose_3d_preoptimization = np.concatenate([current_frame_init[np.newaxis,:,:],self.optimized_poses[:-1,:,:]])

    def get_error(self):
        return 0