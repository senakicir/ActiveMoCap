from helpers import *
import pandas as pd
import torch
import numpy as np
from State import HUMAN_POS_IND, DRONE_POS_IND, DRONE_ORIENTATION_IND, L_SHOULDER_IND, R_SHOULDER_IND
from crop import Crop
from square_bounding_box import *
from kalman_filters import *

class PoseEstimationClient(object):
    def __init__(self, model):
        self.FLIGHT_WINDOW_SIZE =6
        self.CALIBRATION_LENGTH =35

        self.error_2d = []
        self.error_3d = []
        self.requiredEstimationData = []
        self.poseList_3d = []
        self.poseList_3d_calibration = []

        self.liftPoseList = []
        self.requiredEstimationData_calibration = []
        self.isCalibratingEnergy = True
        self.boneLengths = 0
        self.lr = 0
        self.mu = 0
        self.iter_3d = 0
        self.weights = {}
        self.model = model
        self.cropping_tool = Crop()
        self.M = read_M(model)
        #self.kalman = ExtendedKalman()#Kalman()
        self.measurement_cov = np.zeros([1,1])

    def reset(self):
        self.error_2d = []
        self.error_3d = []
        self.isCalibratingEnergy = True
        return 0

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
