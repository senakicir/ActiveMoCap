from helpers import *
import pandas as pd
import torch
import numpy as np
from State import HUMAN_POS_IND, DRONE_POS_IND, DRONE_ORIENTATION_IND, L_SHOULDER_IND, R_SHOULDER_IND
from crop import Crop
from square_bounding_box import *
from kalman_filters import *

class PoseEstimationClient(object):
    def __init__(self, param):
        self.modes = param["MODES"]
        self.method = param["METHOD"]
        self.ftol = param["FTOL"]
        #pose_client.kalman.init_process_noise(kalman_arguments["KALMAN_PROCESS_NOISE_AMOUNT"])
        self.model = param["MODEL"]
        _, _, num_of_joints, _ = model_settings(self.model)

        self.boneLengths = torch.zeros([num_of_joints-1,1])
        self.FLIGHT_WINDOW_SIZE = param["FLIGHT_WINDOW_SIZE"]
        self.CALIBRATION_LENGTH = param["CALIBRATION_LENGTH"]

        self.plot_info = []
        self.error_2d = []
        self.error_3d = []
        self.poseList_3d = []
        self.poseList_3d_calibration = []
        self.liftPoseList = []

        self.requiredEstimationData = []
        self.requiredEstimationData_calibration = []

        self.calib_res_list = []
        self.flight_res_list = []
        self.f_string = ""
        self.processing_time = []

        self.isCalibratingEnergy = True

        self.cropping_tool = Crop()
        self.param_read_M = param["PARAM_READ_M"]
        if self.param_read_M:
            self.param_find_M = False
        else:
            self.param_find_M = param["PARAM_FIND_M"]
        self.calc_hess = param["CALCULATE_HESSIAN"]
        
        self.future_pose = np.zeros([3, num_of_joints])

        if self.param_read_M:
            self.M = read_M(self.model, "M_rel")
        else:
            self.M = np.eye(num_of_joints)

        #self.kalman = ExtendedKalman()#Kalman()
        self.measurement_cov = np.eye(3)
        self.future_measurement_cov = np.eye(3)

        self.calib_cov_list = []
        self.flight_cov_list = []
        
        self.quiet = param["QUIET"]

        self.result_shape_calib = [3, num_of_joints]
        self.result_shape_future = [3, num_of_joints]
        self.result_shape_flight = [self.FLIGHT_WINDOW_SIZE+1, 3, num_of_joints]

        self.weights_calib = {"proj":0.8, "sym":0.2}
        self.weights_flight = param["WEIGHTS"]
        self.weights_future = {"proj":0.8, "bone":0.2}

        self.loss_dict_calib = CALIBRATION_LOSSES
        self.loss_dict_flight = LOSSES
        self.loss_dict_future = FUTURE_LOSSES

    def reset(self, plot_loc):
        if self.param_find_M:
            M = find_M(self.flight_res_list, self.model)
            #plot_matrix(M, plot_loc, 0, "M", "M")
       
        self.plot_info = []
        self.error_2d = []
        self.error_3d = []
        self.requiredEstimationData = []
        self.poseList_3d = []
        self.poseList_3d_calibration = []
        self.liftPoseList = []
        self.requiredEstimationData_calibration = []
        
        self.calib_cov_list = []
        self.flight_cov_list = []

        self.isCalibratingEnergy = True
        return 0        

    def append_res(self, new_res):
        self.processing_time.append(new_res["eval_time"])
        self.f_string = new_res["f_string"]
        if self.isCalibratingEnergy:
            self.calib_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})
        else:
            self.flight_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})

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
            self.flight_cov_list.append({"curr":self.measurement_cov ,"future":self.future_measurement_cov})

    def changeCalibrationMode(self, calibMode):
        self.isCalibratingEnergy = calibMode

    def addNewCalibrationFrame(self, pose_2d, R_drone, C_drone, pose3d_):
        self.requiredEstimationData_calibration.insert(0, [pose_2d, R_drone, C_drone])
        if (len(self.requiredEstimationData_calibration) > self.CALIBRATION_LENGTH):
            self.requiredEstimationData_calibration.pop()
        self.poseList_3d_calibration = pose3d_

    def addNewFrame(self, pose_2d, R_drone, C_drone, pose3d_, pose3d_lift = None):
        self.requiredEstimationData.insert(0, [pose_2d, R_drone, C_drone])
        if (len(self.requiredEstimationData) > self.FLIGHT_WINDOW_SIZE):
            self.requiredEstimationData.pop()

        self.poseList_3d.insert(0, pose3d_)
        if (len(self.poseList_3d) > self.FLIGHT_WINDOW_SIZE):
            self.poseList_3d.pop()

        self.liftPoseList.insert(0, pose3d_lift)
        if (len(self.liftPoseList) > self.FLIGHT_WINDOW_SIZE):
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
