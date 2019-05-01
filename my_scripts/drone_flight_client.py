from helpers import *
import pandas as pd
import torch
import numpy as np

class DroneFlightClient(object):
    def __init__(self, filenames):
        #take filenames and save them
        self.linecount = 0
        self.is_using_airsim = False
        self.end = False

        self.drone_pos_matrix = pd.read_csv(filenames["f_drone_pos"], sep='\t').as_matrix().astype('float')       
        self.groundtruth_matrix = pd.read_csv(filenames["f_groundtruth"], sep='\t').as_matrix().astype('float')       
        self.DRONE_INITIAL_POS = pd.read_csv(filenames["f_initial_drone_pos"], sep='\t').as_matrix().astype('float')       

        self.num_of_data = self.drone_pos_matrix.shape[0]

    def simPauseHuman(self, arg1):
        pass

    def simPauseDrone(self, arg1):
        pass

    def simSetCameraOrientation(self, arg1, arg2):
        pass

    def moveToPositionAsync(self):
        pass

    def read_frame_gt_values(self):
        bone_pos_gt_flat = self.groundtruth_matrix[self.linecount, :]
        drone_transformation_flat = self.drone_pos_matrix[self.linecount, :]

        bone_pos_gt = bone_pos_gt_flat.reshape((3,-1))
        drone_transformation_matrix = drone_transformation_flat.reshape((3,3))
        return bone_pos_gt, drone_transformation_matrix

    def simGetImages(self, arg):
        return DummyPhotoResponse()

    def reset(self):
        self.linecount = 0
        self.end = False

    def changeAnimation(self, arg):
        pass

    def simSetVehiclePose(self, *args):
        pass    
    

class DummyPhotoResponse(object):
    def __init__(self):
        self.bone_pos = np.array([])
        self.image_data_uint8 = np.uint8(0)

