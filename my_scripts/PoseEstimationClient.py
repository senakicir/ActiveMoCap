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
        self.weights = param["WEIGHTS"]
    #pose_client.kalman.init_process_noise(kalman_arguments["KALMAN_PROCESS_NOISE_AMOUNT"])
        self.model = param["MODEL"]

        if self.model =="mpi":
            self.boneLengths = torch.zeros([14,1])
        else:
            self.boneLengths = torch.zeros([20,1])

        self.FLIGHT_WINDOW_SIZE = param["FLIGHT_WINDOW_SIZE"]
        self.CALIBRATION_LENGTH = param["CALIBRATION_LENGTH"]

        self.plot_info = []
        self.error_2d = []
        self.error_3d = []
        self.requiredEstimationData = []
        self.poseList_3d = []
        self.poseList_3d_calibration = []
        self.liftPoseList = []
        self.requiredEstimationData_calibration = []

        self.calib_res_list = []
        self.flight_res_list = []
        self.f_string = ""
        self.processing_time = []

        self.isCalibratingEnergy = True

        self.cropping_tool = Crop()
        self.param_read_M = param["PARAM_READ_M"]
        self.param_find_M = param["PARAM_FIND_M"]

        if self.param_read_M:
            self.M = read_M(self.model, "M_rel")
        else:
            _, _, num_of_joints, _ = model_settings(self.model)
            self.M = np.eye(num_of_joints)
        #self.kalman = ExtendedKalman()#Kalman()
        self.measurement_cov = np.zeros([1,1])
        
        self.quiet = param["QUIET"]


    def reset(self, plot_loc):
        if not self.param_find_M:
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

        self.isCalibratingEnergy = True
        return 0        

    def append_res(self, new_res):
        self.processing_time.append(new_res["eval_time"])
        self.f_string = new_res["f_string"]
        if self.isCalibratingEnergy:
            self.calib_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})
        else:
            self.flight_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})

    def updateMeasurementCov(self, cov):
        self.measurement_cov = cov

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
