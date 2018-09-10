from helpers import *
import pandas as pd
import torch
import numpy as np
from State import HUMAN_POS_IND, DRONE_POS_IND, DRONE_ORIENTATION_IND, L_SHOULDER_IND, R_SHOULDER_IND

class PoseEstimationClient(object):
    def __init__(self, model):
        self.WINDOW_SIZE = 6
        self.error_2d = []
        self.error_3d = []
        self.requiredEstimationData = []
        self.poseList_3d = []
        self.liftPoseList = []
        self.requiredEstimationData_calibration = []
        self.poseList_3d_calibration = []
        self.end = False
        self.isCalibratingEnergy = True
        self.boneLengths = 0
        self.lr = 0
        self.mu = 0
        self.iter_3d = 0
        self.weights = {}
        self.model = ""
        self.cropping_tool = Crop()

    def get_model_info():
        return = 0

    def reset(self):
        self.error_2d = []
        self.error_3d = []
        return 0

    def changeCalibrationMode(self, calibMode):
        self.isCalibratingEnergy = calibMode

    def addNewCalibrationFrame(self, pose_2d, R_drone, C_drone, pose3d_):
        self.requiredEstimationData_calibration.insert(0, [pose_2d, R_drone, C_drone])
        if (len(self.requiredEstimationData_calibration) > CALIBRATION_LENGTH):
            self.requiredEstimationData_calibration.pop()
        
        self.poseList_3d_calibration.insert(0, pose3d_)
        if (len(self.poseList_3d_calibration) > CALIBRATION_LENGTH):
            self.poseList_3d_calibration.pop()

    def addNewFrame(self, pose_2d, R_drone, C_drone, pose3d_, pose3d_lift = None):
        self.requiredEstimationData.insert(0, [pose_2d, R_drone, C_drone])
        if (len(self.requiredEstimationData) > self.WINDOW_SIZE):
            self.requiredEstimationData.pop()

        self.poseList_3d.insert(0, pose3d_)
        if (len(self.poseList_3d) > self.WINDOW_SIZE):
            self.poseList_3d.pop()

        self.liftPoseList.insert(0, pose3d_lift)
        if (len(self.liftPoseList) > self.WINDOW_SIZE):
            self.liftPoseList.pop()

    def update3dPos(self, pose3d_, all = False):
        if (all):
            for ind in range(0,len(self.poseList_3d)):
                self.poseList_3d[ind] = pose3d_
        else:
            self.poseList_3d[0] = pose3d_

    def choose10Frames(self, start_frame):
        unreal_positions_list = []
        bone_pos_3d_GT_list = []
        for frame_ind in range(start_frame, start_frame+self.WINDOW_SIZE):
            response = self.simGetImages(frame_ind)
            unreal_positions_list.append(response.unreal_positions)
            bone_pos_3d_GT_list.append(response.bone_pos)

        return unreal_positions_list, bone_pos_3d_GT_list

