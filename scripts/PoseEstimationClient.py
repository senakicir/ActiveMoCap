import pandas as pd
import torch
import numpy as np

from crop import Basic_Crop
from square_bounding_box import BoundingBox
from Projection_Client import Projection_Client
from Lift_Client import Lift_Client
from pose_helper_functions import calculate_bone_lengths, calculate_bone_lengths_sqrt, add_noise_to_pose, model_settings

class PoseEstimationClient(object):
    def __init__(self, param, general_param, intrinsics):
        self.param = param
        self.general_param = general_param
        self.simulate_error_mode = False
        self.intrinsics = intrinsics

        self.modes = param["MODES"]
        self.method = param["METHOD"]
        self.model  = param["MODEL"]
        self.ftol = param["FTOL"]
        self.xtol = param["XTOL"]

        self.device = torch.device("cpu")
        self.loop_mode = general_param["LOOP_MODE"]
        self.animation = general_param["ANIMATION_NUM"]
        self.is_calibrating_energy = general_param["CALIBRATION_MODE"]

        self.bone_connections, self.joint_names, self.num_of_joints = model_settings(self.model)
        self.hip_index = self.joint_names.index("spine1") #mpi only

        self.ESTIMATION_WINDOW_SIZE = param["ESTIMATION_WINDOW_SIZE"]
        self.FUTURE_WINDOW_SIZE = param["FUTURE_WINDOW_SIZE"]
        self.ONLINE_WINDOW_SIZE = self.ESTIMATION_WINDOW_SIZE+self.FUTURE_WINDOW_SIZE 

        self.CURRENT_POSE_INDEX = self.FUTURE_WINDOW_SIZE
        self.MIDDLE_POSE_INDEX = self.FUTURE_WINDOW_SIZE+(self.ESTIMATION_WINDOW_SIZE)//2
        self.IMMEDIATE_FUTURE_POSE_INDEX = self.FUTURE_WINDOW_SIZE -1
        self.PASTMOST_POSE_INDEX = self.ONLINE_WINDOW_SIZE-1
        

        self.CALIBRATION_WINDOW_SIZE = param["CALIBRATION_WINDOW_SIZE"]
        self.PREDEFINED_MOTION_MODE_LENGTH = param["PREDEFINED_MOTION_MODE_LENGTH"]
        self.save_errors_after = self.PREDEFINED_MOTION_MODE_LENGTH-1

        self.quiet = param["QUIET"]
        self.INIT_POSE_MODE = param["INIT_POSE_MODE"]

        self.USE_SYMMETRY_TERM = param["USE_SYMMETRY_TERM"]
        self.SMOOTHNESS_MODE = param["SMOOTHNESS_MODE"]
        self.USE_LIFT_TERM = param["USE_LIFT_TERM"]
        self.USE_BONE_TERM = param["USE_BONE_TERM"]
        self.BONE_LEN_METHOD = param["BONE_LEN_METHOD"]
        self.PROJECTION_METHOD = param["PROJECTION_METHOD"]
        self.LIFT_METHOD = param["LIFT_METHOD"]
        self.NUMBER_OF_TRAJ_PARAM = param["NUMBER_OF_TRAJ_PARAM"]
        self.NOISE_3D_INIT_STD = param["NOISE_3D_INIT_STD"]


        self.weights_calib =  {key: torch.tensor(value).float() for key, value in param["WEIGHTS_CALIB"].items()}
        self.weights_online = {key: torch.tensor(value).float() for key, value in param["WEIGHTS"].items()}
        self.weights_smooth = self.weights_online["smooth"]
        self.weights_future = {key: torch.tensor(value).float() for key, value in param["WEIGHTS_FUTURE"].items()}

        self.loss_dict_calib = ["proj"]
        if self.USE_SYMMETRY_TERM:  
            self.loss_dict_calib.append("sym")
        self.loss_dict_online = ["proj", "smooth"]
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
            self.result_shape = [self.ONLINE_WINDOW_SIZE, 3, self.num_of_joints]

        self.result_size= np.prod(np.array(self.result_shape))

        self.optimized_traj = np.zeros([self.NUMBER_OF_TRAJ_PARAM, 3, self.num_of_joints])

        self.plot_info = []
        
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
        self.poses_3d_gt_debugger = np.zeros([10, 3, self.num_of_joints])

        self.boneLengths = torch.zeros([self.num_of_joints-1]).to(self.device)
        self.batch_bone_lengths = (self.boneLengths).repeat(self.ONLINE_WINDOW_SIZE,1)
        self.multiple_bone_lengths = torch.zeros([self.ONLINE_WINDOW_SIZE, self.num_of_joints-1])

        self.lift_pose_tensor = torch.zeros([self.ESTIMATION_WINDOW_SIZE, 3, self.num_of_joints]).to(self.device)
        self.pose_2d_tensor = torch.zeros([self.ESTIMATION_WINDOW_SIZE, 2, self.num_of_joints]).to(self.device)
        self.pose_2d_gt_tensor = torch.zeros([self.ESTIMATION_WINDOW_SIZE, 2, self.num_of_joints]).to(self.device)
        self.cam_list = []
        self.potential_projected_est = torch.zeros([2, self.num_of_joints]).to(self.device)

        self.requiredEstimationData = []

        self.calib_res_list = []
        self.online_res_list = []
        self.processing_time = []

        self.current_pose = np.zeros([3, self.num_of_joints])
        self.middle_pose = np.zeros([3, self.num_of_joints])
        self.future_poses = np.zeros([self.FUTURE_WINDOW_SIZE, 3, self.num_of_joints])
        self.immediate_future_pose =  np.zeros([3, self.num_of_joints])

        self.adj_current_pose = np.zeros([3, self.num_of_joints])
        self.adj_middle_pose = np.zeros([3, self.num_of_joints])
        self.adj_future_poses = np.zeros([self.FUTURE_WINDOW_SIZE, 3, self.num_of_joints])

        self.projection_client = Projection_Client(test_set=self.animation, future_window_size=self.FUTURE_WINDOW_SIZE, 
                                                    estimation_window_size=self.ESTIMATION_WINDOW_SIZE, 
                                                    num_of_joints=self.num_of_joints, intrinsics=intrinsics, 
                                                    device=self.device)
        self.lift_client = Lift_Client(self.ESTIMATION_WINDOW_SIZE, self.FUTURE_WINDOW_SIZE)

        self.SIZE_X, self.SIZE_Y = intrinsics["size_x"],  intrinsics["size_y"]

        self.margin = 0.2
        self.cropping_tool = Basic_Crop(margin=self.margin)

    def model_settings(self):
        return self.bone_connections, self.joint_names, self.num_of_joints, self.hip_index

    def reset_crop(self, loop_mode):
        if self.cropping_tool is not None:
            self.cropping_tool = Basic_Crop(margin=self.margin)

    def reset(self, plot_loc):
        pass

    def read_bone_lengths_from_file(self, file_manager, pose_3d_gt):
        assert not self.is_calibrating_energy
        if self.modes["bone_len"] == "calib_res":
            if self.animation == "noise":
                raise NotImplementedError
            if file_manager.bone_lengths_dict is not None:
                self.boneLengths[:] = torch.from_numpy(file_manager.bone_lengths_dict[self.BONE_LEN_METHOD])
                self.batch_bone_lengths = (self.boneLengths).repeat(self.ONLINE_WINDOW_SIZE,1)
        
        elif self.modes["bone_len"] == "gt":
            use_bones = torch.from_numpy(pose_3d_gt.copy())
            bone_connections = np.array(self.bone_connections)
            if self.BONE_LEN_METHOD == "no_sqrt":
                current_bone_lengths = calculate_bone_lengths(bones=use_bones, bone_connections=bone_connections, batch=False)
            elif self.BONE_LEN_METHOD == "sqrt":
                current_bone_lengths = calculate_bone_lengths_sqrt(bones=use_bones, bone_connections=bone_connections, batch=False)
            self.boneLengths = current_bone_lengths.clone()
            self.batch_bone_lengths = (self.boneLengths).repeat(self.ONLINE_WINDOW_SIZE,1)


    def update_bone_lengths(self, bones):
        assert self.is_calibrating_energy
        bone_connections = np.array(self.bone_connections)
        use_bones = bones.clone()
        current_bone_lengths_no_sqrt = calculate_bone_lengths(bones=use_bones, bone_connections=bone_connections, batch=False)
        current_bone_lengths_sqrt = calculate_bone_lengths_sqrt(bones=use_bones, bone_connections=bone_connections, batch=False)
        if self.animation == "noise":
            raise NotImplementedError
            self.multiple_bone_lengths = torch.cat((current_bone_lengths.unsqueeze(0),self.multiple_bone_lengths[:-1,:]), dim=0)
        else:
            self.boneLengths = {}
            self.boneLengths["no_sqrt"] = current_bone_lengths_no_sqrt.clone()
            self.boneLengths["sqrt"] = current_bone_lengths_sqrt.clone()

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
        return self.errors

    def addNewFrame(self, linecount, pose_2d, pose_2d_gt, inv_transformation_matrix, pose3d_lift, current_pose_3d_gt, futuremost_pose_3d_gt, camera_id):
        self.requiredEstimationData.insert(0, [camera_id, pose_2d.clone(), pose_2d_gt.clone(), inv_transformation_matrix.clone()])
        self.poses_3d_gt_debugger = np.concatenate([current_pose_3d_gt[np.newaxis, :, :], self.poses_3d_gt_debugger[0:-1]], axis=0)
        if linecount > 2 and not self.is_calibrating_energy:
            assert np.mean(np.std(self.poses_3d_gt_debugger, axis=0)) != 0 

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
            #if linecount > self.ONLINE_WINDOW_SIZE:
                #assert not np.allclose(self.poses_3d_gt[self.CURRENT_POSE_INDEX+1], self.poses_3d_gt[-1])

            fail_msg = "The distance between two consequtive gt values are: " + str(np.linalg.norm(current_pose_3d_gt[:,self.hip_index]-self.poses_3d_gt[self.CURRENT_POSE_INDEX-1,:,self.hip_index]))
            #assert np.linalg.norm(current_pose_3d_gt[:,self.hip_index]-self.poses_3d_gt[self.CURRENT_POSE_INDEX-1,:,self.hip_index])<2, fail_msg
            # print(pose3d_lift.unsqueeze(0).shape)
            # print( self.lift_pose_tensor[:-1].shape)
            # self.lift_pose_tensor[1:] = torch.cat((pose3d_lift.unsqueeze(0), self.lift_pose_tensor[:-1]), dim=0)
            temp = self.lift_pose_tensor[:-1,:,:].clone() 
            self.lift_pose_tensor[0,:,: ] = pose3d_lift.clone()
            self.lift_pose_tensor[1:,:,:] = temp.clone()
            # self.pose_2d_tensor[1:] =  torch.cat([pose_2d.unsqueeze(0), self.pose_2d_tensor[:-1], dim=0)
            # self.pose_2d_gt_tensor[1:] =  torch.cat([pose_2d_gt.unsqueeze(0), self.pose_2d_gt_tensor[:-1], dim=0)
            # self.cam_list.insert(0, camera_id)

            # self.inv_transformation_tensor[1:] = torch.cat([inv_transformation_matrix.unsqueeze(0), self.inv_transformation_tensor[:-1], dim=0)

        
        if self.is_calibrating_energy:
            if linecount >= self.PREDEFINED_MOTION_MODE_LENGTH:
                while len(self.requiredEstimationData) > self.CALIBRATION_WINDOW_SIZE:
                    self.requiredEstimationData.pop()
        else:
            while len(self.requiredEstimationData) > self.ESTIMATION_WINDOW_SIZE:
                self.requiredEstimationData.pop()

    def init_frames(self, pose_2d, pose_2d_gt, inv_transformation_matrix, pose3d_lift, pose_3d_gt, future_poses_3d_gt, camera_id):
        self.poses_3d_gt[self.FUTURE_WINDOW_SIZE:, :, :] = np.repeat(pose_3d_gt[np.newaxis, :, :].copy(), self.ESTIMATION_WINDOW_SIZE, axis=0)
        self.poses_3d_gt[:self.FUTURE_WINDOW_SIZE, :, :] = future_poses_3d_gt.copy()
        self.poses_3d_gt_debugger[:,:,:] = pose_3d_gt.copy()
        if self.is_calibrating_energy:
            self.requiredEstimationData.insert(0, [camera_id, pose_2d.clone(), pose_2d_gt.clone(), inv_transformation_matrix.clone()])
        else:
            for _ in range(self.ESTIMATION_WINDOW_SIZE):
                self.requiredEstimationData.insert(0, [camera_id, pose_2d.clone(), pose_2d_gt.clone(), inv_transformation_matrix.clone()])
            if self.USE_LIFT_TERM:
                self.lift_pose_tensor[:, :, :] = pose3d_lift.clone()

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
        new_pose_client = PoseEstimationClient(self.param, self.general_param, self.intrinsics)

        new_pose_client.projection_client = self.projection_client.deepcopy_projection_client()
        new_pose_client.lift_client = self.lift_client.deepcopy_lift_client()

        new_pose_client.requiredEstimationData = []
        for camera_id, bone_2d, bone_2d_gt, inverse_transformation_matrix in self.requiredEstimationData:
            new_bone_2d = bone_2d.clone()
            new_bone_2d_gt = bone_2d_gt.clone()
            new_inverse_transformation_matrix = inverse_transformation_matrix.clone()
            new_pose_client.requiredEstimationData.append([camera_id, new_bone_2d, new_bone_2d_gt, new_inverse_transformation_matrix])

        for key, error_list in self.errors.items():
            new_pose_client.errors[key] = error_list.copy()
        new_pose_client.average_errors = self.average_errors.copy()
        new_pose_client.overall_error = self.overall_error
        new_pose_client.ave_overall_error = self.ave_overall_error

        new_pose_client.optimized_poses = self.optimized_poses.copy()
        new_pose_client.adjusted_optimized_poses = self.adjusted_optimized_poses.copy()
        new_pose_client.pose_3d_preoptimization = self.pose_3d_preoptimization.copy()

        new_pose_client.poses_3d_gt = self.poses_3d_gt.copy()
        new_pose_client.poses_3d_gt_debugger = self.poses_3d_gt_debugger.copy()

        new_pose_client.boneLengths = self.boneLengths.clone()
        new_pose_client.batch_bone_lengths = self.batch_bone_lengths.clone()
        new_pose_client.multiple_bone_lengths = self.multiple_bone_lengths.clone()

        new_pose_client.lift_pose_tensor = self.lift_pose_tensor.clone()
        new_pose_client.potential_projected_est = self.potential_projected_est.clone()

        new_pose_client.future_poses = self.future_poses.copy()
        new_pose_client.current_pose = self.current_pose.copy()
        new_pose_client.middle_pose = self.middle_pose.copy()
        new_pose_client.immediate_future_pose = self.immediate_future_pose.copy()

        new_pose_client.adj_future_poses = self.adj_future_poses.copy()
        new_pose_client.adj_current_pose = self.adj_current_pose.copy()
        new_pose_client.adj_middle_pose = self.adj_middle_pose.copy()

        if self.cropping_tool is not None:
            new_pose_client.cropping_tool = self.cropping_tool.copy_cropping_tool()

        new_pose_client.quiet = True
        new_pose_client.simulate_error_mode = False
        new_pose_client.trial_ind = trial_ind

        return new_pose_client