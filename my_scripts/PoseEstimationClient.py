from helpers import *
import pandas as pd
import torch
import numpy as np
from crop import Crop
from square_bounding_box import *
from project_bones import Projection_Client
from Lift_Client import Lift_Client
from pose_helper_functions import calculate_bone_lengths, calculate_bone_lengths_sqrt, add_noise_to_pose

class PoseEstimationClient(object):
    def __init__(self, param, general_param, intrinsics_focal, intrinsics_px, intrinsics_py, image_size):
        self.param = param
        self.general_param = general_param
        self.simulate_error_mode = False

        self.modes = param["MODES"]
        self.method = param["METHOD"]
        self.model  = param["MODEL"]
        self.ftol = param["FTOL"]
        self.xtol = param["XTOL"]
        self.USE_SINGLE_JOINT = param["USE_SINGLE_JOINT"]

        #self.device = torch.device(('cpu', 'cuda')[general_param["run_loc"] == "server"])
        self.device = torch.device("cpu")
        self.loop_mode = general_param["LOOP_MODE"]
        self.animation = general_param["ANIMATION_NUM"]

        self.bone_connections, self.joint_names, self.num_of_joints = model_settings(self.model)
        if self.USE_SINGLE_JOINT:
            self.num_of_joints = 1
            self.hip_index = 0
        else:
            self.hip_index = self.joint_names.index("spine1") #mpi only

        self.ESTIMATION_WINDOW_SIZE = param["ESTIMATION_WINDOW_SIZE"]
        self.FUTURE_WINDOW_SIZE = param["FUTURE_WINDOW_SIZE"]
        self.ONLINE_WINDOW_SIZE = self.ESTIMATION_WINDOW_SIZE+self.FUTURE_WINDOW_SIZE 

        self.CURRENT_POSE_INDEX = self.FUTURE_WINDOW_SIZE
        self.MIDDLE_POSE_INDEX = self.FUTURE_WINDOW_SIZE+(self.ESTIMATION_WINDOW_SIZE)//2
        self.FUTURE_POSE_INDEX = self.FUTURE_WINDOW_SIZE -1
        self.PASTMOST_POSE_INDEX = self.ONLINE_WINDOW_SIZE-1
        

        self.CALIBRATION_WINDOW_SIZE = param["CALIBRATION_WINDOW_SIZE"]
        self.PREDEFINED_MOTION_MODE_LENGTH = param["PREDEFINED_MOTION_MODE_LENGTH"]
        
        self.quiet = param["QUIET"]
        self.INIT_POSE_MODE = param["INIT_POSE_MODE"]
        self.NOISE_2D_STD = param["NOISE_2D_STD"]
        self.NOISE_LIFT_STD = param["NOISE_LIFT_STD"]
        self.NOISE_3D_INIT_STD = param["NOISE_3D_INIT_STD"]
        self.USE_SYMMETRY_TERM = param["USE_SYMMETRY_TERM"]
        self.SMOOTHNESS_MODE = param["SMOOTHNESS_MODE"]
        self.USE_LIFT_TERM = param["USE_LIFT_TERM"]
        self.USE_BONE_TERM = param["USE_BONE_TERM"]
        self.BONE_LEN_METHOD = param["BONE_LEN_METHOD"]
        self.PROJECTION_METHOD = param["PROJECTION_METHOD"]
        self.LIFT_METHOD = param["LIFT_METHOD"]
        self.USE_TRAJECTORY_BASIS = param["USE_TRAJECTORY_BASIS"]
        self.NUMBER_OF_TRAJ_PARAM = param["NUMBER_OF_TRAJ_PARAM"]


        if self.loop_mode == "calibration":
            self.is_calibrating_energy = True
        else:
            self.is_calibrating_energy = False

        self.optimized_traj = np.zeros([self.NUMBER_OF_TRAJ_PARAM, 3, self.num_of_joints])

        self.plot_info = []
        self.error_2d = []
        
        self.errors = {}
        self.average_errors = {}
        for index in range(self.ONLINE_WINDOW_SIZE):
            self.errors[index] = []
            self.average_errors[index] = -1
        self.overall_error = -1
        self.ave_overall_error = -1
        
        self.openpose_error = 0
        self.openpose_arm_error = 0
        self.openpose_leg_error = 0

        self.optimized_poses = np.zeros([self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints])
        self.adjusted_optimized_poses = np.zeros([self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints])
        self.pose_3d_preoptimization = np.zeros([self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints])

        self.poses_3d_gt = np.zeros([self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints])

        self.boneLengths = torch.zeros([self.num_of_joints-1]).to(self.device)
        self.batch_bone_lengths = (self.boneLengths).repeat(self.ONLINE_WINDOW_SIZE,1)
        self.multiple_bone_lengths = torch.zeros([self.ONLINE_WINDOW_SIZE, self.num_of_joints-1])

        self.lift_pose_tensor = torch.zeros([self.ESTIMATION_WINDOW_SIZE, 3, self.num_of_joints]).to(self.device)
        self.potential_projected_est = torch.zeros([2, self.num_of_joints]).to(self.device)


        self.requiredEstimationData = []

        self.calib_res_list = []
        self.online_res_list = []
        self.processing_time = []

        self.param_read_M = param["PARAM_READ_M"]
        if self.param_read_M:
            self.param_find_M = False
        else:
            self.param_find_M = param["PARAM_FIND_M"]

        self.current_pose = np.zeros([3, self.num_of_joints])
        self.middle_pose = np.zeros([3, self.num_of_joints])
        self.future_poses = np.zeros([self.FUTURE_WINDOW_SIZE, 3, self.num_of_joints])
        self.immediate_future_pose =  np.zeros([3, self.num_of_joints])

        self.adj_current_pose = np.zeros([3, self.num_of_joints])
        self.adj_middle_pose = np.zeros([3, self.num_of_joints])
        self.adj_future_poses = np.zeros([self.FUTURE_WINDOW_SIZE, 3, self.num_of_joints])

        if self.param_read_M:
            self.M = read_M(self.num_of_joints, "M_rel")
        else:
            self.M = np.eye(self.num_of_joints)

        self.weights_calib = param["WEIGHTS_CALIB"]
        self.weights_online = param["WEIGHTS"]
        self.weights_smooth = self.weights_online["smooth"]
        self.weights_future = param["WEIGHTS_FUTURE"]

        self.loss_dict_calib = ["proj"]
        if self.USE_SYMMETRY_TERM:  
            self.loss_dict_calib.append("sym")
        self.loss_dict_online = ["proj", "smooth"]
        if not self.USE_SINGLE_JOINT:
            if self.USE_BONE_TERM:
                self.loss_dict_online.append("bone")
            if self.USE_LIFT_TERM:
                self.loss_dict_online.append("lift")
        self.loss_dict = {}

        if self.is_calibrating_energy:
            self.result_shape = [3, self.num_of_joints]
            self.loss_dict = self.loss_dict_calib
        else:
            self.loss_dict = self.loss_dict_online
            if self.USE_TRAJECTORY_BASIS:
                self.result_shape = [self.NUMBER_OF_TRAJ_PARAM, 3, self.num_of_joints]
            else:
                self.result_shape = [self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints]
        self.result_size =  np.prod(np.array(self.result_shape))

        self.intrinsics_focal = intrinsics_focal
        self.intrinsics_px = intrinsics_px
        self.intrinsics_py = intrinsics_py
        self.SIZE_X, self.SIZE_Y = image_size

        self.projection_client = Projection_Client(test_set=self.animation, future_window_size=self.FUTURE_WINDOW_SIZE, num_of_joints=self.num_of_joints, focal_length=self.intrinsics_focal, px=self.intrinsics_px, py=self.intrinsics_py, noise_2d_std=self.NOISE_2D_STD, device=self.device)
        self.lift_client = Lift_Client(self.NOISE_LIFT_STD)

        if self.animation == "drone_flight":
            self.cropping_tool = None
        else:
            self.cropping_tool = Crop(loop_mode = self.loop_mode, size_x=self.SIZE_X, size_y= self.SIZE_Y)

        self.save_errors_after = self.PREDEFINED_MOTION_MODE_LENGTH-1

    def model_settings(self):
        return self.bone_connections, self.joint_names, self.num_of_joints, self.hip_index

    def reset_crop(self, loop_mode):
        if self.animation == "drone_flight":
            self.cropping_tool = None
        else:
            self.cropping_tool = Crop(loop_mode = loop_mode, size_x=self.SIZE_X, size_y= self.SIZE_Y)

    def reset(self, plot_loc):
        if self.param_find_M:
            M = find_M(plot_info=self.online_res_list, joint_names=self.joint_names, num_of_joints=self.num_of_joints)
            #plot_matrix(M, plot_loc, 0, "M", "M")
        return 0   

    def read_bone_lengths_from_file(self, file_manager):
        if self.animation == "noise":
            raise NotImplementedError
        self.boneLengths[:] = torch.from_numpy(file_manager.f_bone_len_array)
        self.batch_bone_lengths = (self.boneLengths).repeat(self.ONLINE_WINDOW_SIZE,1)

    def update_bone_lengths(self, bones):
        if self.modes["bone_len"] == "calib_res":
            use_bones = bones.clone()
        elif self.modes["bone_len"] == "gt":
            use_bones = torch.from_numpy(self.poses_3d_gt[0,:,:].copy())

        bone_connections = np.array(self.bone_connections)
        if self.BONE_LEN_METHOD == "no_sqrt":
            current_bone_lengths = calculate_bone_lengths(bones=use_bones, bone_connections=bone_connections, batch=False)
        elif self.BONE_LEN_METHOD == "sqrt":
            current_bone_lengths = calculate_bone_lengths_sqrt(bones=use_bones, bone_connections=bone_connections, batch=False)
       
        if self.animation == "noise":
            self.multiple_bone_lengths = torch.cat((current_bone_lengths.unsqueeze(0),self.multiple_bone_lengths[:-1,:]), dim=0)
        else:
            self.boneLengths = current_bone_lengths
            self.batch_bone_lengths = (self.boneLengths).repeat(self.ONLINE_WINDOW_SIZE,1)

    def append_res(self, new_res):
        self.processing_time.append(new_res["eval_time"])
        if self.is_calibrating_energy:
            self.calib_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})
        else:
            self.online_res_list.append({"est":  new_res["est"], "GT": new_res["GT"], "drone": new_res["drone"]})

    def calculate_store_errors(self, linecount):
        if linecount > self.save_errors_after:
            sum_all_errors = 0
            for index in range(self.ONLINE_WINDOW_SIZE):
                error_calc = np.mean(np.linalg.norm(self.poses_3d_gt[index, :, :] 
                                    - self.adjusted_optimized_poses[index, :, :], axis=0))
                self.errors[index].append(error_calc)
                sum_all_errors += error_calc
                self.average_errors[index] = sum(self.errors[index])/len(self.errors[index])
            
            self.overall_error = sum_all_errors/self.ONLINE_WINDOW_SIZE
            self.ave_overall_error = sum(self.average_errors.values())/self.ONLINE_WINDOW_SIZE
            #print(self.average_errors)
        return self.errors

    def addNewFrame(self, linecount, pose_2d, pose_2d_gt, inv_transformation_matrix, pose3d_lift, current_pose_3d_gt, futuremost_pose_3d_gt):
        self.requiredEstimationData.insert(0, [pose_2d.clone(), pose_2d_gt.clone(), inv_transformation_matrix.clone()])

        if self.is_calibrating_energy:
            self.poses_3d_gt[:,:,:] = current_pose_3d_gt.copy()
        else:
            old_gt = self.poses_3d_gt.copy()
            self.poses_3d_gt = np.concatenate([futuremost_pose_3d_gt[np.newaxis, :, :], self.poses_3d_gt[0:-1]], axis=0)
            assert np.allclose(self.poses_3d_gt[1:], old_gt[:-1])
            self.poses_3d_gt[self.CURRENT_POSE_INDEX] = current_pose_3d_gt.copy()
            assert np.allclose(self.poses_3d_gt[0], futuremost_pose_3d_gt)
            assert np.allclose(self.poses_3d_gt[1:self.CURRENT_POSE_INDEX], old_gt[0:self.CURRENT_POSE_INDEX - 1])
            assert np.allclose(self.poses_3d_gt[self.CURRENT_POSE_INDEX + 1:], old_gt[self.CURRENT_POSE_INDEX:-1])
            assert np.allclose(self.poses_3d_gt[self.CURRENT_POSE_INDEX], current_pose_3d_gt)
            if linecount > self.ONLINE_WINDOW_SIZE:
                assert not np.allclose(self.poses_3d_gt[self.CURRENT_POSE_INDEX+1], self.poses_3d_gt[-1])

            print("currentpose",current_pose_3d_gt[:,0])
            print(self.poses_3d_gt[:,:,0])
            print("*****")

            temp = self.lift_pose_tensor[:-1,:,:].clone() 
            self.lift_pose_tensor[0,:,: ] = pose3d_lift.clone()
            self.lift_pose_tensor[1:,:,:] = temp.clone()
        
        if self.is_calibrating_energy:
            if linecount >= self.PREDEFINED_MOTION_MODE_LENGTH:
                while len(self.requiredEstimationData) > self.CALIBRATION_WINDOW_SIZE:
                    self.requiredEstimationData.pop()
        else:
            while len(self.requiredEstimationData) > self.ESTIMATION_WINDOW_SIZE:
                self.requiredEstimationData.pop()

    def init_frames(self, pose_2d, pose_2d_gt, inv_transformation_matrix, pose3d_lift, pose_3d_gt, future_poses_3d_gt):
        self.poses_3d_gt[self.FUTURE_WINDOW_SIZE:, :, :] = np.repeat(pose_3d_gt[np.newaxis, :, :].copy(), self.ESTIMATION_WINDOW_SIZE, axis=0)
        self.poses_3d_gt[:self.FUTURE_WINDOW_SIZE, :, :] = future_poses_3d_gt.copy()
        if self.is_calibrating_energy:
            self.requiredEstimationData.insert(0, [pose_2d.clone(), pose_2d_gt.clone(), inv_transformation_matrix.clone()])
        else:
            for _ in range(self.ESTIMATION_WINDOW_SIZE):
                self.requiredEstimationData.insert(0, [pose_2d.clone(), pose_2d_gt.clone(), inv_transformation_matrix.clone()])
            self.lift_pose_tensor[:, :, :] = pose3d_lift.clone()

    def set_initial_pose_old(self, linecount, pose_3d_gt, pose_2d, transformation_matrix):
        if (linecount != 0):
            current_frame_init = self.future_poses[0,:,:].copy() #futuremost pose
        else:
            if self.INIT_POSE_MODE == "gt":
                current_frame_init = pose_3d_gt.copy()
            elif self.INIT_POSE_MODE == "gt_with_noise":
                current_frame_init = add_noise_to_pose(torch.from_numpy(pose_3d_gt), self.NOISE_3D_INIT_STD).numpy()
            elif self.INIT_POSE_MODE == "backproj":
                current_frame_init = self.projection_client.take_single_backprojection(pose_2d, transformation_matrix, self.joint_names).numpy()
            elif self.INIT_POSE_MODE == "zeros":
                current_frame_init = np.zeros([3, self.num_of_joints])

        if self.is_calibrating_energy:
            self.pose_3d_preoptimization = current_frame_init.copy()
        else:
            self.pose_3d_preoptimization = np.concatenate([current_frame_init[np.newaxis,:,:],self.optimized_poses[:-1,:,:]])

    def set_initial_pose(self):
        current_frame_init = self.future_poses[0,:,:].copy() #futuremost pose

        if self.is_calibrating_energy:
            self.pose_3d_preoptimization = current_frame_init.copy()
        else:
            self.pose_3d_preoptimization = np.concatenate([current_frame_init[np.newaxis,:,:],self.optimized_poses[:-1,:,:]])
           

    def update3dPos(self, optimized_poses, adjusted_optimized_poses):
        if (self.is_calibrating_energy):
            self.current_pose = optimized_poses.copy()
            self.middle_pose = optimized_poses.copy()
            self.future_poses = np.repeat(optimized_poses[np.newaxis, :, :].copy(), self.FUTURE_WINDOW_SIZE, axis=0)
            self.immediate_future_pose =  optimized_poses.copy()

            self.adj_current_pose = adjusted_optimized_poses.copy()
            self.adj_middle_pose = adjusted_optimized_poses.copy()
            self.adj_future_poses = np.repeat(adjusted_optimized_poses[np.newaxis, :, :].copy(), self.FUTURE_WINDOW_SIZE, axis=0)

            self.optimized_poses = np.repeat(optimized_poses[np.newaxis, :, :].copy(), self.ONLINE_WINDOW_SIZE, axis=0)
            self.adjusted_optimized_poses = np.repeat(adjusted_optimized_poses[np.newaxis, :, :].copy(), self.ONLINE_WINDOW_SIZE, axis=0)

        else:
            self.current_pose =  optimized_poses[self.CURRENT_POSE_INDEX, :,:].copy() #current pose
            self.middle_pose = optimized_poses[self.MIDDLE_POSE_INDEX, :,:].copy() #middle_pose
            self.future_poses = optimized_poses[:self.FUTURE_WINDOW_SIZE, :,:].copy() #future_poses
            self.immediate_future_pose =  optimized_poses[self.FUTURE_WINDOW_SIZE-1, :,:].copy() #immediate future pose

            self.adj_current_pose =  adjusted_optimized_poses[self.CURRENT_POSE_INDEX, :,:].copy() #current pose
            self.adj_middle_pose = adjusted_optimized_poses[self.MIDDLE_POSE_INDEX, :,:].copy() #middle_pose
            self.adj_future_poses = adjusted_optimized_poses[:self.FUTURE_WINDOW_SIZE, :,:].copy() #future_poses

            self.optimized_poses = optimized_poses.copy()
            self.adjusted_optimized_poses = adjusted_optimized_poses.copy()

    def deepcopy_PEC(self, trial_ind):
        new_pose_client = PoseEstimationClient(self.param, self.general_param, self.intrinsics_focal, self.intrinsics_px, self.intrinsics_py, (self.SIZE_X, self.SIZE_Y))

        new_pose_client.projection_client = self.projection_client.deepcopy_projection_client()
        new_pose_client.lift_client = self.lift_client.deepcopy_lift_client()

        new_pose_client.optimized_poses = self.optimized_poses.copy()
        new_pose_client.adjusted_optimized_poses = self.adjusted_optimized_poses.copy()
        new_pose_client.pose_3d_preoptimization = self.pose_3d_preoptimization.copy()

        new_pose_client.requiredEstimationData = []
        for bone_2d, bone_2d_gt, inverse_transformation_matrix in self.requiredEstimationData:
            new_bone_2d = bone_2d.clone()
            new_bone_2d_gt = bone_2d_gt.clone()
            new_inverse_transformation_matrix = inverse_transformation_matrix.clone()
            new_pose_client.requiredEstimationData.append([new_bone_2d, new_bone_2d_gt, new_inverse_transformation_matrix])

        new_pose_client.lift_pose_tensor = self.lift_pose_tensor.clone()
        new_pose_client.poses_3d_gt = self.poses_3d_gt.copy()
        new_pose_client.boneLengths = self.boneLengths.clone()
        new_pose_client.batch_bone_lengths = self.batch_bone_lengths.clone()
        new_pose_client.multiple_bone_lengths = self.multiple_bone_lengths.clone()

        new_pose_client.error_2d = self.error_2d.copy()
        for key, error_list in self.errors.items():
            new_pose_client.errors[key] = error_list.copy()
        new_pose_client.average_errors = self.average_errors.copy()

        new_pose_client.future_poses = self.future_poses.copy()
        new_pose_client.current_pose = self.current_pose.copy()
        new_pose_client.middle_pose = self.middle_pose.copy()
        new_pose_client.immediate_future_pose = self.immediate_future_pose.copy()

        new_pose_client.adj_future_poses = self.adj_future_poses.copy()
        new_pose_client.adj_current_pose = self.adj_current_pose.copy()
        new_pose_client.adj_middle_pose = self.adj_middle_pose.copy()

        new_pose_client.potential_projected_est = self.potential_projected_est.clone()

        if self.animation != "drone_flight":
            new_pose_client.cropping_tool = self.cropping_tool.copy_cropping_tool()

        new_pose_client.quiet = True
        new_pose_client.simulate_error_mode = False
        new_pose_client.trial_ind = trial_ind
        new_pose_client.is_calibrating_energy = self.is_calibrating_energy
        new_pose_client.result_shape, new_pose_client.result_size = self.result_shape, self.result_size

        return new_pose_client