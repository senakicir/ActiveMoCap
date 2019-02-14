from helpers import *
import pandas as pd
import torch
import numpy as np
from crop import Crop
from square_bounding_box import *
from kalman_filters import *

class PoseEstimationClient(object):
    def __init__(self, param, cropping_tool):
        self.modes = param["MODES"]
        self.method = param["METHOD"]
        self.model  = param["MODEL"]
        self.ftol = param["FTOL"]
        self.bone_connections, self.joint_names, self.num_of_joints = model_settings(self.model)

        self.ONLINE_WINDOW_SIZE = param["ONLINE_WINDOW_SIZE"]
        self.CALIBRATION_WINDOW_SIZE = param["CALIBRATION_WINDOW_SIZE"]
        self.CALIBRATION_LENGTH = param["CALIBRATION_LENGTH"]
        self.PRECALIBRATION_LENGTH = param["PRECALIBRATION_LENGTH"]
        self.quiet = param["QUIET"]
        self.init_pose_with_gt = param["INIT_POSE_WITH_GT"]

        self.numpy_random = np.random.RandomState(param["SEED"])
        torch.manual_seed(param["SEED"])

        self.plot_info = []
        self.error_2d = []
        self.error_3d = []
        self.middle_pose_error = []
        self.openpose_error = 0
        self.openpose_arm_error = 0
        self.openpose_leg_error = 0


        self.poseList_3d = []
        self.poseList_3d_calibration = []
        self.liftPoseList = []
        self.boneLengths = torch.zeros([self.num_of_joints-1,1])

        self.requiredEstimationData = []
        self.requiredEstimationData_calibration = []

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
        self.future_pose = np.zeros([3, self.num_of_joints])
        self.P_world =  0 #all 6 poses

        if self.param_read_M:
            self.M = read_M(self.model, "M_rel")
        else:
            self.M = np.eye(self.num_of_joints)

        #self.kalman = ExtendedKalman()#Kalman()
        self.measurement_cov = np.eye(3)
        self.future_measurement_cov = np.eye(3)

        self.calib_cov_list = []
        self.online_cov_list = []

        self.result_shape_calib = [3, self.num_of_joints]
        self.result_shape_online = [self.ONLINE_WINDOW_SIZE+1, 3, self.num_of_joints]

        self.weights_calib = {"proj":0.8, "sym":0.2}
        self.weights_online = param["WEIGHTS"]
        self.weights_future = {'proj': 0.33, 'smooth': 0.33, 'bone': 0.33}#param["WEIGHTS"]

        self.loss_dict_calib = CALIBRATION_LOSSES
        self.loss_dict_online = ONLINE_LOSSES
        self.loss_dict_future = FUTURE_LOSSES

#        self.cam_pitch = 0 #move to state
        self.middle_pose_GT_list = []

        self.f_string = ""
        self.f_reconst_string = ""
        self.f_groundtruth_str = ""

    def reset_crop(self, loop_mode):
        self.cropping_tool = Crop(loop_mode=loop_mode)

    def reset(self, plot_loc):
        if self.param_find_M:
            M = find_M(self.online_res_list, self.model)
            #plot_matrix(M, plot_loc, 0, "M", "M")
       
        self.isCalibratingEnergy = True
        return 0        

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

    def addNewCalibrationFrame(self, pose_2d, R_drone, C_drone, R_cam, pose3d_, linecount):
        #pre calibration!
        if linecount < self.PRECALIBRATION_LENGTH:
            self.requiredEstimationData_calibration.insert(0, [pose_2d, R_drone, C_drone, R_cam])
        else:
            self.requiredEstimationData_calibration.insert(0, [pose_2d, R_drone, C_drone, R_cam])
            while len(self.requiredEstimationData_calibration) > self.CALIBRATION_WINDOW_SIZE:
                self.requiredEstimationData_calibration.pop()
        self.poseList_3d_calibration = pose3d_

    def addNewFrame(self, pose_2d, R_drone, C_drone, R_cam, pose3d_, pose3d_lift = None):
        self.requiredEstimationData.insert(0, [pose_2d, R_drone, C_drone, R_cam])
        if (len(self.requiredEstimationData) > self.ONLINE_WINDOW_SIZE):
            self.requiredEstimationData.pop()

        self.poseList_3d.insert(0, pose3d_)
        if (len(self.poseList_3d) > self.ONLINE_WINDOW_SIZE):
            self.poseList_3d.pop()

        self.liftPoseList.insert(0, pose3d_lift)
        if (len(self.liftPoseList) > self.ONLINE_WINDOW_SIZE):
            self.liftPoseList.pop()

    def update3dPos(self, pose3d_, is_calib = False):
        if (is_calib):
            self.poseList_3d_calibration = pose3d_
            for ind in range(0,len(self.poseList_3d)):
                self.poseList_3d[ind] = pose3d_.copy()
        else:
            for ind in range(0,len(self.poseList_3d)):
                self.poseList_3d[ind] = pose3d_[ind, :, :].copy()


    def update3dPos_2(self, pose3d_, is_calib = False):
        if (is_calib):
            for ind in range(0,len(self.poseList_3d_calibration)):
                self.poseList_3d_calibration[ind] = pose3d_.copy()
        else:
            for ind in range(0,len(self.poseList_3d)):
                self.poseList_3d[ind] = pose3d_[ind, :, :].copy()

    def update_middle_pose_GT(self, middle_pose):
        self.middle_pose_GT_list.insert(0, middle_pose)
        if (len(self.middle_pose_GT_list) > MIDDLE_POSE_INDEX):
            self.middle_pose_GT_list.pop()
        return self.middle_pose_GT_list[-1]