from helpers import *
import pandas as pd
import torch
import numpy as np
from crop import Crop
from square_bounding_box import *
from project_bones import take_bone_backprojection_pytorch

def calculate_bone_lengths(bones, bone_connections, batch):
    if batch:
        return (torch.sum(torch.pow(bones[:, :, bone_connections[:,0]] - bones[:, :, bone_connections[:,1]], 2), dim=1))
    else:  
        return (torch.sum(torch.pow(bones[:, bone_connections[:,0]] - bones[:, bone_connections[:,1]], 2), dim=0))


class PoseEstimationClient(object):
    def __init__(self, param, cropping_tool):
        self.simulate_error_mode = False

        self.modes = param["MODES"]
        self.method = param["METHOD"]
        self.model  = param["MODEL"]
        self.ftol = param["FTOL"]
        self.USE_SINGLE_JOINT = param["USE_SINGLE_JOINT"]
        self.bone_connections, self.joint_names, self.num_of_joints = model_settings(self.model)
        if self.USE_SINGLE_JOINT:
            self.num_of_joints = 1
            self.hip_index = 0
        else:
            self.hip_index = self.joint_names.index("spine1") #mpi only

        self.ONLINE_WINDOW_SIZE = param["ONLINE_WINDOW_SIZE"]
        self.CALIBRATION_WINDOW_SIZE = param["CALIBRATION_WINDOW_SIZE"]
        self.CALIBRATION_LENGTH = param["CALIBRATION_LENGTH"]
        self.PRECALIBRATION_LENGTH = param["PRECALIBRATION_LENGTH"]
        self.quiet = param["QUIET"]
        self.init_pose_with_gt = param["INIT_POSE_WITH_GT"]
        self.noise_2d_std = param["NOISE_2D_STD"]
        self.USE_SYMMETRY_TERM = param["USE_SYMMETRY_TERM"]
        self.SMOOTHNESS_MODE = param["SMOOTHNESS_MODE"]
        self.USE_LIFT_TERM = param["USE_LIFT_TERM"]
        self.USE_BONE_TERM = param["USE_BONE_TERM"]
        self.USE_TRAJECTORY_BASIS = param["USE_TRAJECTORY_BASIS"]
        self.NUMBER_OF_TRAJ_PARAM = param["NUMBER_OF_TRAJ_PARAM"]

        self.optimized_traj = np.zeros([self.NUMBER_OF_TRAJ_PARAM, 3, self.num_of_joints])

        self.numpy_random = np.random.RandomState(param["SEED"])
        torch.manual_seed(param["SEED"])

        self.plot_info = []
        self.error_2d = []
        self.error_3d = []
        self.middle_pose_error = []
        self.openpose_error = 0
        self.openpose_arm_error = 0
        self.openpose_leg_error = 0

        self.optimized_poses = np.zeros([self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints])
        self.pose_3d_preoptimization = np.zeros([self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints])
        self.liftPoseList = []
        self.poses_3d_gt = np.zeros([self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints])
        self.boneLengths = torch.zeros([self.num_of_joints-1,1])

        self.requiredEstimationData = []

        self.calib_res_list = []
        self.online_res_list = []
        self.processing_time = []

        self.isCalibratingEnergy = True

        self.cropping_tool = cropping_tool
        self.param_read_M = param["PARAM_READ_M"]
        if self.param_read_M:
            self.param_find_M = False
        else:
            self.param_find_M = param["PARAM_FIND_M"]

        self.current_pose = np.zeros([3, self.num_of_joints])
        self.middle_pose = np.zeros([3, self.num_of_joints])
        self.future_pose = np.zeros([3, self.num_of_joints])

        self.result_shape = [3, self.num_of_joints]
        self.result_size = np.prod(np.array(self.result_shape))

        if self.param_read_M:
            self.M = read_M(self.num_of_joints, "M_rel")
        else:
            self.M = np.eye(self.num_of_joints)

        #self.kalman = ExtendedKalman()#Kalman()
        self.measurement_cov = np.eye(3)
        self.future_measurement_cov = np.eye(3)

        self.calib_cov_list = []
        self.online_cov_list = []
        self.middle_pose_GT_list = []

        self.weights_calib = {"proj":0.8, "sym":0.2}
        self.weights_online = param["WEIGHTS"]

        self.loss_dict_calib = ["proj"]
        if self.USE_SYMMETRY_TERM:  
            self.loss_dict_calib.append("sym")
        self.loss_dict_online = ["proj", "smooth"]
        if not self.USE_SINGLE_JOINT:
            if self.USE_BONE_TERM:
                self.loss_dict_online.append("bone")
            if self.USE_LIFT_TERM:
                self.loss_dict_online.append("lift")
        self.loss_dict = {}


        self.f_string = ""
        self.f_reconst_string = ""
        self.f_groundtruth_str = ""


    def model_settings(self):
        return self.bone_connections, self.joint_names, self.num_of_joints, self.hip_index

    def reset_crop(self, loop_mode):
        self.cropping_tool = Crop(loop_mode=loop_mode)

    def reset(self, plot_loc):
        if self.param_find_M:
            M = find_M(plot_info=self.online_res_list, joint_names=self.joint_names, num_of_joints=self.num_of_joints)
            #plot_matrix(M, plot_loc, 0, "M", "M")
        self.isCalibratingEnergy = True
        return 0   

    def update_bone_lengths(self, bones):
        bone_connections = np.array(self.bone_connections)
        self.boneLengths = calculate_bone_lengths(bones=bones, bone_connections=bone_connections, batch=False)

    def append_res(self, new_res):
        self.processing_time.append(new_res["eval_time"])
        self.f_string = new_res["f_string"]
        if self.isCalibratingEnergy:
            self.calib_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})
        else:
            self.online_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})

    def updateMeasurementCov(self, cov, curr_pose_ind, future_pose_ind):
        if  self.isCalibratingEnergy:
            curr_pose_ind = 0
        curr_inv_hess = shape_cov(cov, self.model, curr_pose_ind)
        self.measurement_cov = curr_inv_hess
        if self.isCalibratingEnergy:
            self.calib_cov_list.append(self.measurement_cov)
        else:
            future_inv_hess = shape_cov(cov, self.model, future_pose_ind)
            self.future_measurement_cov = future_inv_hess
            self.online_cov_list.append({"curr":self.measurement_cov ,"future":self.future_measurement_cov})

    def updateMeasurementCov_mini(self, cov, curr_pose_ind, future_pose_ind):
        if self.isCalibratingEnergy:
            curr_pose_ind = 0
        curr_inv_hess = shape_cov_mini(cov, self.model, curr_pose_ind)
        self.measurement_cov = curr_inv_hess
        if self.isCalibratingEnergy:
            self.calib_cov_list.append(self.measurement_cov)
        else:
            future_inv_hess = shape_cov_mini(cov, self.model, future_pose_ind)
            self.future_measurement_cov = future_inv_hess
            self.online_cov_list.append({"curr":self.measurement_cov ,"future":self.future_measurement_cov})

    def changeCalibrationMode(self, calibMode):
        self.isCalibratingEnergy = calibMode
        if self.isCalibratingEnergy:
            self.result_shape = [3, self.num_of_joints]
            self.loss_dict = self.loss_dict_calib
        else:
            self.loss_dict = self.loss_dict_online
            if self.USE_TRAJECTORY_BASIS:
                self.result_shape = [self.NUMBER_OF_TRAJ_PARAM, 3, self.num_of_joints]
            else:
                self.result_shape = [self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints]
        self.result_size =  np.prod(np.array(self.result_shape))

    def addNewFrame(self, pose_2d, pose_2d_gt, R_drone, C_drone, R_cam, linecount, pose_3d_gt, pose3d_lift):
        self.liftPoseList.insert(0, pose3d_lift)
        self.requiredEstimationData.insert(0, [pose_2d, pose_2d_gt, R_drone, C_drone, R_cam])

        temp = self.poses_3d_gt[:-1,:].copy() 
        self.poses_3d_gt[0,:] = pose_3d_gt.copy()
        self.poses_3d_gt[1:,:] = temp.copy()
        
        if self.isCalibratingEnergy:
            if linecount >= self.PRECALIBRATION_LENGTH:
                while len(self.requiredEstimationData) > self.CALIBRATION_WINDOW_SIZE-1:
                    self.requiredEstimationData.pop()
                    self.liftPoseList.pop()

        else:
            if (len(self.requiredEstimationData) > self.ONLINE_WINDOW_SIZE-1):
                self.requiredEstimationData.pop()
                self.liftPoseList.pop()
                
    def update3dPos(self, optimized_poses):
        if (self.isCalibratingEnergy):
            self.current_pose = optimized_poses.copy()
            self.middle_pose = optimized_poses.copy()
            self.future_pose = optimized_poses.copy()
            self.optimized_poses = np.repeat(optimized_poses[np.newaxis, :, :], self.ONLINE_WINDOW_SIZE, axis=0).copy()
        else:
            self.current_pose =  optimized_poses[CURRENT_POSE_INDEX, :,:].copy() #current pose
            self.middle_pose = optimized_poses[MIDDLE_POSE_INDEX, :,:].copy() #middle_pose
            self.future_pose =  optimized_poses[FUTURE_POSE_INDEX, :,:].copy() #future pose
            self.optimized_poses = optimized_poses.copy()
        
    def update_middle_pose_GT(self, middle_pose):
        self.middle_pose_GT_list.insert(0, middle_pose)
        if (len(self.middle_pose_GT_list) > MIDDLE_POSE_INDEX):
            self.middle_pose_GT_list.pop()
        return self.middle_pose_GT_list[-1]

    def set_initial_pose(self, linecount, pose_3d_gt, pose_2d, R_drone_gt, C_drone_gt, R_cam_gt):
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