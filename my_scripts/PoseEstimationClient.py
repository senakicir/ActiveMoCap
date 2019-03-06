from helpers import *
import pandas as pd
import torch
import numpy as np
from crop import Crop
from square_bounding_box import *
from kalman_filters import *

def calculate_bone_lengths(bones, bone_connections, batch):
    if batch:
        return (torch.sum(torch.pow(bones[:, :, bone_connections[:,0]] - bones[:, :, bone_connections[:,1]], 2), dim=1))
    else:  
        return (torch.sum(torch.pow(bones[:, bone_connections[:,0]] - bones[:, bone_connections[:,1]], 2), dim=0))

def calculate_bone_directions(bones, lift_bone_directions, batch):
    if batch:
        current_bone_vector = bones[1:, :, lift_bone_directions[:,0]] - bones[1:, :, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=1, keepdim=True)).repeat(1,3,1) #try without repeat
    else:
        current_bone_vector = bones[:, lift_bone_directions[:,0]] - bones[:, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=0, keepdim=True)).repeat(3,1) #try without repeat
    return current_bone_vector/(norm_bone_vector+EPSILON)

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
        self.noise_2d_std = param["NOISE_2D_STD"]
        self.USE_SYMMETRY_TERM = param["USE_SYMMETRY_TERM"]
        self.USE_SINGLE_JOINT = param["USE_SINGLE_JOINT"]
        self.SMOOTHNESS_MODE = param["SMOOTHNESS_MODE"]
        self.USE_LIFT_TERM = param["USE_LIFT_TERM"]

        self.numpy_random = np.random.RandomState(param["SEED"])
        torch.manual_seed(param["SEED"])

        self.plot_info = []
        self.error_2d = []
        self.error_3d = []
        self.middle_pose_error = []
        self.openpose_error = 0
        self.openpose_arm_error = 0
        self.openpose_leg_error = 0

        self.P_world = np.zeros([self.ONLINE_WINDOW_SIZE+1, 3, self.num_of_joints])
        self.liftPoseList = []
        self.thrownAway_liftPoseList = None
        self.thrownAway_requiredEstimationData = None
        self.threw_Away = True
        
        self.boneLengths = torch.zeros([self.num_of_joints-1,1])

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
        self.future_pose = np.zeros([3, self.num_of_joints])

        self.prev_poseList_3d = []
        self.result_shape = [3, self.num_of_joints]

        if self.param_read_M:
            self.M = read_M(self.model, "M_rel")
        else:
            self.M = np.eye(self.num_of_joints)

        #self.kalman = ExtendedKalman()#Kalman()
        self.measurement_cov = np.eye(3)
        self.future_measurement_cov = np.eye(3)

        self.calib_cov_list = []
        self.online_cov_list = []

        self.weights_calib = {"proj":0.8, "sym":0.2}
        self.weights_online = param["WEIGHTS"]
        self.weights_future = param["WEIGHTS"]

        self.loss_dict_calib = ["proj"]
        if self.USE_SYMMETRY_TERM:  
            self.loss_dict_calib.append("sym")
        self.loss_dict_online = ["proj", "smooth", "bone", "lift"]
        self.loss_dict_future = ["proj", "smooth", "bone", "lift"]

#        self.cam_pitch = 0 #move to state
        self.middle_pose_GT_list = []

        self.f_string = ""
        self.f_reconst_string = ""
        self.f_groundtruth_str = ""

        self.noise_2d = 0

    def reset_crop(self, loop_mode):
        self.cropping_tool = Crop(loop_mode=loop_mode)

    def reset(self, plot_loc):
        if self.param_find_M:
            M = find_M(self.online_res_list, self.model)
            #plot_matrix(M, plot_loc, 0, "M", "M")
        self.isCalibratingEnergy = True
        return 0   

    def update_bone_lengths(self, bones):
        bone_connections = np.array(self.bone_connections)
        self.boneLengths = calculate_bone_lengths(bones=bones, bone_connections=bone_connections, batch=False)

    def append_res(self, new_res):
        self.processing_time.append(new_res["eval_time"])
        self.f_string = new_res["f_string"]
        if self.isCalibratingEnergy:
            self.calib_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})
        else:
            self.online_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})

    def add_2d_noise(self, bone_2d):
        self.noise_2d = torch.normal(torch.zeros(bone_2d.shape), torch.ones(bone_2d.shape)*self.noise_2d_std)
        bone_2d = bone_2d.clone() + self.noise_2d
        return bone_2d 

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
        else:
            self.result_shape = [self.ONLINE_WINDOW_SIZE+1, 3, self.num_of_joints]

    def rewind_step(self):
        self.middle_pose_GT_list.pop(0)
        middle_pose_error = self.middle_pose_error.pop()

        self.requiredEstimationData.pop(0)
        self.liftPoseList.pop(0)
        if self.threw_Away:
            self.liftPoseList.append(self.thrownAway_liftPoseList)
            self.requiredEstimationData.append(self.thrownAway_requiredEstimationData)

        self.P_world = self.prev_P_world.copy()

        error = self.error_3d.pop()
        self.error_2d.pop()

        if self.isCalibratingEnergy:
            self.calib_res_list.pop()
        else:
            self.online_res_list.pop()
        return error 

    def addNewFrame(self, pose_2d, R_drone, C_drone, R_cam, linecount, pose3d_lift = None):
        self.threw_Away = False
        self.liftPoseList.insert(0, pose3d_lift)
        self.requiredEstimationData.insert(0, [pose_2d, R_drone, C_drone, R_cam])
        if self.isCalibratingEnergy:
            if linecount >= self.PRECALIBRATION_LENGTH:
                while len(self.requiredEstimationData) > self.CALIBRATION_WINDOW_SIZE:
                    self.thrownAway_requiredEstimationData = self.requiredEstimationData.pop()
                    self.thrownAway_liftPoseList = self.liftPoseList.pop()
                    self.threw_Away = True

        else:
            if (len(self.requiredEstimationData) > self.ONLINE_WINDOW_SIZE):
                self.thrownAway_requiredEstimationData = self.requiredEstimationData.pop()
                self.thrownAway_liftPoseList = self.liftPoseList.pop()
                self.threw_Away = True
        

    def update3dPos(self, P_world):
        self.prev_P_world = self.P_world.copy()

        if (self.isCalibratingEnergy):
            self.current_pose = P_world.copy()
            self.middle_pose = P_world.copy()
            self.future_pose = P_world.copy()
            self.P_world = np.repeat(P_world[np.newaxis, :, :], self.ONLINE_WINDOW_SIZE+1, axis=0).copy()
        else:
            self.current_pose =  P_world[CURRENT_POSE_INDEX, :,:].copy() #current pose
            self.middle_pose = P_world[MIDDLE_POSE_INDEX, :,:].copy() #middle_pose
            self.future_pose =  P_world[FUTURE_POSE_INDEX, :,:].copy() #future pose
            self.P_world = P_world.copy()

        
    def update_middle_pose_GT(self, middle_pose):
        self.middle_pose_GT_list.insert(0, middle_pose)
        if (len(self.middle_pose_GT_list) > MIDDLE_POSE_INDEX):
            self.middle_pose_GT_list.pop()
        return self.middle_pose_GT_list[-1]