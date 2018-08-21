from helpers import *
import pandas as pd
import torch
import numpy as np
from State import HUMAN_POS_IND, DRONE_POS_IND, DRONE_ORIENTATION_IND, L_SHOULDER_IND, R_SHOULDER_IND

class NonAirSimClient(object):
    def __init__(self, filename_bones, filename_others):
        groundtruth_matrix = pd.read_csv(filename_bones, sep='\t', header=None).ix[:,1:].as_matrix().astype('float')                
        self.DRONE_INITIAL_POS = groundtruth_matrix[0,0:3]
        self.WINDOW_SIZE = 6
        self.groundtruth = groundtruth_matrix[1:,:-1]
        a_flight_matrix = pd.read_csv(filename_others, sep='\t', header=None).ix
        self.a_flight = a_flight_matrix[:,1:].as_matrix().astype('float')
        self.linecount = 0
        self.current_bone_pos = 0
        self.current_unreal_pos = 0
        self.current_drone_pos = airsim.Vector3r()
        self.current_drone_orient = 0
        self.num_of_data = self.a_flight.shape[0]
        self.error_2d = []
        self.error_3d = []
        self.requiredEstimationData = []
        self.poseList_3d = []
        self.liftPoseList = []
        self.requiredEstimationData_calibration = []
        self.poseList_3d_calibration = []
        self.end = False
        self.isCalibratingEnergy = True
        self.boneLengths = CALIBRATION_LENGTH
        self.lr = 0
        self.mu = 0
        self.iter_3d = 0
        self.weights = {}
        self.model = ""

    def moveToPositionAsync(self, arg1, arg2, arg3, arg4, arg5, arg6, arg7, yaw_or_rate=0 ,lookahead=0, adaptive_lookahead=0):
        if (self.linecount == self.num_of_data-1):
            self.end = True

    def getPosition(self):
        position = airsim.Vector3r()
        (position.x_val, position.y_val, position.z_val) = self.a_flight[self.linecount, 6:9]
        return position

    def getPitchRollYaw(self):
        (pitch, roll, yaw) = self.a_flight[self.linecount, 3:6]
        return (pitch, roll, yaw)

    def updateSynchronizedData(self, unreal_positions_, bone_positions_, drone_pos_, drone_orient_):
        self.current_bone_pos = np.copy(bone_positions_)
        self.current_unreal_pos = np.copy(unreal_positions_)
        self.current_drone_pos = drone_pos_
        self.current_drone_orient = np.copy(drone_orient_)
        return 0
    
    def getSynchronizedData(self):
        return self.current_unreal_pos, self.current_bone_pos, self.current_drone_pos, self.current_drone_orient

    def simGetImages(self, which_frame = None):
        if which_frame == None:
            frame_num = self.linecount
        else:
            frame_num = which_frame

        response = prepareDataForResponse(self.groundtruth, self.DRONE_INITIAL_POS, frame_num)

        return response

    def reset(self):
        self.error_2d = []
        self.error_3d = []
        return 0

    def changeAnimation(self, newAnim):
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

class DummyPhotoResponse(object):
    bone_pos = np.array([])
    unreal_positions = np.zeros([5,3])
    image_data_uint8 = np.uint8(0)

def prepareDataForResponse(data, initial_pos, linecount):
    response = DummyPhotoResponse()
    X = np.copy(data[linecount, :])
    line = np.reshape(X, (-1, 3))
    response.bone_pos = line[3:,:].T
    keys = {DRONE_POS_IND: 0, DRONE_ORIENTATION_IND: 1, HUMAN_POS_IND: 2}
    for key, value in keys.items():
        response.unreal_positions[key, :] = line[value, :] #dronepos
        if (key != DRONE_ORIENTATION_IND):
            response.unreal_positions[key, 2] = -response.unreal_positions[key, 2] #dronepos
            response.unreal_positions[key, :] = (response.unreal_positions[key, :] - initial_pos)/100
    response.bone_pos[2, :] = -response.bone_pos[2, :] 
    response.bone_pos = (response.bone_pos - initial_pos[:, np.newaxis])/100

    return response
