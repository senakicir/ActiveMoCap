import pandas as pd
import torch
import numpy as np
from PotentialStatesFetcher import PotentialState
from Lift_Client import calculate_bone_directions


class External_Dataset_Client(object):
    def __init__(self, length_of_simulation, test_set_name):
        #take filenames and save them
        self.linecount = 0
        self.online_linecount = 0
        self.default_initial_anim_time = 0
        self.internal_anim_time = self.default_initial_anim_time
        
        self.chosen_cam_view = 0
        self.length_of_simulation = length_of_simulation

        self.is_using_airsim = False
        self.end = False
        self.test_set_name = test_set_name

        self.external_dataset_states = []
        self.DRONE_INITIAL_POS = np.zeros([3])     

    def simPause(self, arg1, arg2):
        pass

    def simSetCameraOrientation(self, arg1, arg2): 
        pass

    def moveToPositionAsync(self, sth):
        pass

    def setAnimationTime(self, anim_time):
        self.internal_anim_time = anim_time

    def getAnimationTime(self):
        return self.internal_anim_time


    def increment_linecount(self, is_calibrating_energy):
        self.linecount += 1
        if not is_calibrating_energy:
            self.online_linecount += 1
        print('linecount:', self.linecount, ', online linecount:', self.online_linecount)

    def get_linecount(self):
        return self.linecount

    def simGetImages(self, arg):
        return [DummyPhotoResponse()]

    def simSetVehiclePose(self, goal_state):
        self.chosen_cam_view = goal_state.index

    def changeAnimation(self, arg):
        pass

    def reset(self):
        self.linecount = 0
        self.online_linecount = 0
        self.chosen_cam_index = 0
        self.end = False


class DummyPhotoResponse(object):
    def __init__(self):
        self.bone_pos = np.array([])
        self.image_data_uint8 = np.uint8(0)

