from helpers import *
import pandas as pd
import torch
import numpy as np
from State import HUMAN_POS_IND, DRONE_POS_IND, DRONE_ORIENTATION_IND, L_SHOULDER_IND, R_SHOULDER_IND

class NonAirSimClient(object):
    def __init__(self, filename_bones, filename_others):
        groundtruth_matrix = pd.read_csv(filename_bones, sep='\t', header=None).ix[:,1:].as_matrix().astype('float')                
        self.DRONE_INITIAL_POS = groundtruth_matrix[0,0:3]
        self.groundtruth = groundtruth_matrix[1:,:-1]
        a_flight_matrix = pd.read_csv(filename_others, sep='\t', header=None).ix
        self.a_flight = a_flight_matrix[:,1:].as_matrix().astype('float')
        self.linecount = 0
        self.current_bone_pos = 0
        self.current_unreal_pos = 0
        self.current_drone_pos = airsim.Vector3r()
        self.current_drone_orient = 0
        self.num_of_data = self.a_flight.shape[0]
        self.end = False

    def moveToPositionAsync(self):
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
    
    def simPause(self, state):
        return 0

    def reset(self):
        return 0

    def changeAnimation(self, newAnim):
        return 0

    def changeCalibrationMode(self, calibMode):
        return 0

class DummyPhotoResponse(object):
    bone_pos = np.array([])
    unreal_positions = np.zeros([5,3])
    image_data_uint8 = np.uint8(0)

def prepareDataForResponse(data, initial_pos, linecount):
    response = DummyPhotoResponse()
    X = np.copy(data[linecount, :])
    line = np.reshape(X, (-1, 3))
    response.bone_pos = line[3:,:].T
    keys = [DRONE_POS_IND, DRONE_ORIENTATION_IND, HUMAN_POS_IND]
    for key in keys:
        response.unreal_positions[key, :] = line[key, :] #dronepos
        if (key != DRONE_ORIENTATION_IND):
            response.unreal_positions[key, 2] = -response.unreal_positions[key, 2] #dronepos
            response.unreal_positions[key, :] = (response.unreal_positions[key, :] - initial_pos)/100   
    response.bone_pos[2, :] = -response.bone_pos[2, :] 
    response.bone_pos = (response.bone_pos - initial_pos[:, np.newaxis])/100

    response.unreal_positions[HUMAN_POS_IND, :] = response.bone_pos[:, joint_names_h36m.index('spine1')]

    return response
