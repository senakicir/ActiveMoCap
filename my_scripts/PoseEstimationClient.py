from helpers import *
import pandas as pd
import torch
import numpy as np
from crop import Crop
from square_bounding_box import *
from project_bones import Projection_Client
from Lift_Client import Lift_Client
from pose_helper_functions import calculate_bone_lengths, calculate_bone_lengths_sqrt, add_noise_to_pose

class PoseEstimationClient(object):
    def __init__(self, param, cropping_tool, animation, intrinsics_focal, intrinsics_px, intrinsics_py):
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
        self.INIT_POSE_MODE = param["INIT_POSE_MODE"]
        self.NOISE_2D_STD = param["NOISE_2D_STD"]
        self.NOISE_LIFT_STD = param["NOISE_LIFT_STD"]
        self.NOISE_3D_INIT_STD = param["NOISE_3D_INIT_STD"]
        self.USE_SYMMETRY_TERM = param["USE_SYMMETRY_TERM"]
        self.SMOOTHNESS_MODE = param["SMOOTHNESS_MODE"]
        self.USE_LIFT_TERM = param["USE_LIFT_TERM"]
        self.USE_BONE_TERM = param["USE_BONE_TERM"]
        self.BONE_LEN_METHOD = param["BONE_LEN_METHOD"]
        self.PROJECTION_METHOD = param["PROJECTION_METHOD"]
        self.LIFT_METHOD = param["LIFT_METHOD"]
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
        self.boneLengths = torch.zeros([self.num_of_joints-1])
        self.multiple_bone_lengths = torch.zeros([self.ONLINE_WINDOW_SIZE, self.num_of_joints-1])

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

        self.animation = animation
        self.projection_client = Projection_Client(test_set=self.animation, num_of_joints=self.num_of_joints, focal_length=intrinsics_focal, px=intrinsics_px, py=intrinsics_py)
        self.lift_client = Lift_Client()

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
        if self.BONE_LEN_METHOD == "no_sqrt":
            current_bone_lengths = calculate_bone_lengths(bones=bones, bone_connections=bone_connections, batch=False)
        elif self.BONE_LEN_METHOD == "sqrt":
            current_bone_lengths = calculate_bone_lengths_sqrt(bones=bones, bone_connections=bone_connections, batch=False)
       
        if self.animation == "noise":
            self.multiple_bone_lengths = torch.cat((current_bone_lengths.unsqueeze(0),self.multiple_bone_lengths[:-1,:]), dim=0)
        else:
            self.boneLengths = current_bone_lengths

    def append_res(self, new_res):
        self.processing_time.append(new_res["eval_time"])
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

    def addNewFrame(self, pose_2d, pose_2d_gt, inv_transformation_matrix, linecount, pose_3d_gt, pose3d_lift):
        self.liftPoseList.insert(0, pose3d_lift)
        self.requiredEstimationData.insert(0, [pose_2d, pose_2d_gt, inv_transformation_matrix])

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

    def set_initial_pose(self, linecount, pose_3d_gt, pose_2d, transformation_matrix):
        if (linecount != 0):
            current_frame_init = self.future_pose.copy()
        else:
            if self.INIT_POSE_MODE == "gt":
                current_frame_init = pose_3d_gt.copy()
            elif self.INIT_POSE_MODE == "gt_with_noise":
                current_frame_init = add_noise_to_pose(pose_3d_gt, self.NOISE_3D_INIT_STD)
            elif self.INIT_POSE_MODE == "backproj":
                current_frame_init = self.projection_client.take_single_backprojection(pose_2d, transformation_matrix, self.joint_names).numpy()
            elif self.INIT_POSE_MODE == "zeros":
                current_frame_init = np.zeros([3, self.num_of_joints])

        if self.isCalibratingEnergy:
            self.pose_3d_preoptimization = current_frame_init.copy()
        else:
            self.pose_3d_preoptimization = np.concatenate([current_frame_init[np.newaxis,:,:],self.optimized_poses[:-1,:,:]])