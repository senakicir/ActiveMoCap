from project_bones import Projection_Client
import torch as torch
from helpers import split_bone_connections, EPSILON, return_lift_bone_connections, euler_to_rotation_matrix, CAMERA_PITCH_OFFSET, CAMERA_ROLL_OFFSET, CAMERA_YAW_OFFSET, CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z
import numpy as np 
from math import pi
from PoseEstimationClient import calculate_bone_lengths, calculate_bone_lengths_sqrt
from Lift_Client import Lift_Client, calculate_bone_directions, calculate_bone_directions_simple

def mse_loss(input_1, input_2, N):
    return torch.sum(torch.pow((input_1 - input_2),2))/N

def weighted_mse_loss(input_1, input_2, weights, N):
    return torch.sum(weights*torch.pow((input_1 - input_2),2))/N    

def blake_zisserman_loss(input_1, input_2):
    N = input_1.data.nelement()
    C = -torch.log(torch.exp(-torch.pow((input_1-input_2),2))+EPSILON)
    return torch.sum(C)/N

def cauchy_loss(input_1, input_2):
    N = input_1.data.nelement()
    b = 1000
    sigma = input_1 - input_2
    C = (b*b)*torch.log(1+torch.pow(sigma,2)/(b*b))
    return torch.sum(C)/N

def find_residuals(input_1, input_2):
    return (torch.pow((input_1 - input_2),2)).view(-1)


class toy_example(torch.nn.Module):
    def __init__(self, model, loss_dict, weights, data_list, M):
        super(toy_example, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([4,1]).to(pose_client.device), requires_grad=True)

    def forward(self):
        return self.pose3d[3]**2 + self.pose3d[2]*self.pose3d[1]*self.pose3d[0] + self.pose3d[0]**4
 
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_calibration_parallel(torch.nn.Module):

    def __init__(self, pose_client, projection_client):
        super(pose3d_calibration_parallel, self).__init__()
        self.bone_connections, _, self.NUM_OF_JOINTS, _ = pose_client.model_settings()
        left_bone_connections, right_bone_connections, _ = split_bone_connections(self.bone_connections)
        self.left_bone_connections = np.array(left_bone_connections)
        self.right_bone_connections = np.array(right_bone_connections)
        self.pose3d = torch.nn.Parameter(torch.zeros(pose_client.result_shape).to(pose_client.device), requires_grad=True)
        self.projection_method = pose_client.PROJECTION_METHOD

        self.energy_weights = pose_client.weights_calib
        self.loss_dict = pose_client.loss_dict_calib
        self.projection_client = projection_client
        self.use_symmetry_term = pose_client.USE_SYMMETRY_TERM

        if self.projection_method == "normal":
            self.projection_scales = 1
        elif self.projection_method == "scaled":
            max_y_vals, _ = torch.max(self.projection_client.pose_2d_tensor[:,1,:], dim=1)
            min_y_vals, _ = torch.min(self.projection_client.pose_2d_tensor[:,1,:], dim=1)
            self.projection_scales = (1/(max_y_vals-min_y_vals))[:, None, None]
        elif self.projection_method == "normalized":
            self.projection_scales = torch.FloatTensor([((1024.0/pose_client.SIZE_X)**2)])
       
        self.pltpts = {}
        self.pltpts_weighted = {}
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
            self.pltpts_weighted[loss_key] = []

    def forward(self):        
        output = {}
        if self.use_symmetry_term :
            left_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, self.left_bone_connections[:,0]] - self.pose3d[:, self.left_bone_connections[:,1]], 2), dim=0))
            right_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, self.right_bone_connections[:,0]] - self.pose3d[:, self.right_bone_connections[:,1]], 2), dim=0))
            bonelosses = torch.pow((left_length_of_bone - right_length_of_bone),2)
            output["sym"] = torch.sum(bonelosses)/6

        rep_pose3d = self.pose3d.repeat(self.projection_client.window_size, 1, 1)
        projected_2d = self.projection_client.take_projection(rep_pose3d)
        output["proj"] = weighted_mse_loss(projected_2d, self.projection_client.pose_2d_tensor, self.projection_scales, (self.projection_client.window_size)*self.NUM_OF_JOINTS)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])
            self.pltpts_weighted[loss_key].append(self.energy_weights[loss_key]*output[loss_key])

        return overall_output

    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_online_parallel(torch.nn.Module):

    def __init__(self, pose_client, projection_client, lift_client, optimization_mode):
        super(pose3d_online_parallel, self).__init__()

        self.optimization_mode = optimization_mode
        assert self.optimization_mode == "estimate_whole" or self.optimization_mode == "estimate_future" or self.optimization_mode == "estimate_past"
        self.animation = pose_client.animation
        self.projection_client = projection_client

        bone_connections, _, self.NUM_OF_JOINTS, self.hip_index = pose_client.model_settings()
        self.bone_connections = np.array(bone_connections)
        self.ESTIMATION_WINDOW_SIZE = pose_client.ESTIMATION_WINDOW_SIZE
        self.FUTURE_WINDOW_SIZE = pose_client.FUTURE_WINDOW_SIZE
        self.ONLINE_WINDOW_SIZE = pose_client.ONLINE_WINDOW_SIZE

#        if self.optimization_mode == "estimate_whole":
        self.pose3d = torch.nn.Parameter(torch.zeros(pose_client.result_shape).to(pose_client.device), requires_grad=True)
       # if self.optimization_mode == "estimate_past":
        #    self.pose3d = torch.nn.Parameter(torch.zeros(pose_client.result_shape_past).to(pose_client.device), requires_grad=True)
        #if self.optimization_mode == "estimate_future"
         #   self.pose3d = torch.nn.Parameter(torch.zeros(pose_client.result_shape_future).to(pose_client.device), requires_grad=True)


        if pose_client.animation == "noise":
            self.bone_lengths = pose_client.multiple_bone_lengths
        else:
            self.batch_bone_lengths = pose_client.batch_bone_lengths

        self.loss_dict = pose_client.loss_dict_online
        
        if self.optimization_mode == "estimate_whole":
            self.energy_weights = pose_client.weights_future
        elif self.optimization_mode == "estimate_past" or self.optimization_mode == "estimate_future":
            self.energy_weights = pose_client.weights_online
   
        self.smoothness_mode = pose_client.SMOOTHNESS_MODE
        self.use_lift_term = pose_client.USE_LIFT_TERM
        self.use_bone_term = pose_client.USE_BONE_TERM
        self.use_single_joint = pose_client.USE_SINGLE_JOINT
        self.bone_len_method = pose_client.BONE_LEN_METHOD
        self.lift_method = pose_client.LIFT_METHOD
        self.projection_method = pose_client.PROJECTION_METHOD

        if self.projection_method == "normal":
            self.projection_scales = 1
        elif self.projection_method == "scaled":
            max_y_vals, _ = torch.max(self.projection_client.pose_2d_tensor[:,1,:], dim=1)
            min_y_vals, _ = torch.min(self.projection_client.pose_2d_tensor[:,1,:], dim=1)
            self.projection_scales = (1/(max_y_vals-min_y_vals))[:, None, None]
        elif self.projection_method == "normalized":
            self.projection_scales = torch.FloatTensor([(1024.0/pose_client.SIZE_X)**2])
       # print("average scale", torch.mean(self.projection_scales))

        if self.use_lift_term and not self.use_single_joint and self.optimization_mode != "estimate_future":
            self.pose3d_lift_directions = lift_client.pose3d_lift_directions
            self.lift_bone_directions = np.array(return_lift_bone_connections(bone_connections))

        self.pltpts = {}
        self.pltpts_weighted = {}
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
            self.pltpts_weighted[loss_key] = []

        self.n = list(range(1, self.ONLINE_WINDOW_SIZE))
        self.n.append(0)
        self.m = list(range(1, self.ONLINE_WINDOW_SIZE-1))
        self.m.append(0)

    def forward(self):
        output = {}

        #projection loss
        if self.optimization_mode == "estimate_future":
            output["proj"]=0
        else:
            if self.optimization_mode == "estimate_past":
                projected_2d = self.projection_client.take_projection(self.pose3d[self.FUTURE_WINDOW_SIZE:])
                proj_N = self.ESTIMATION_WINDOW_SIZE*self.NUM_OF_JOINTS  
            elif self.optimization_mode == "estimate_whole":
                projected_2d = self.projection_client.take_projection(self.pose3d)
                proj_N = self.ONLINE_WINDOW_SIZE*self.NUM_OF_JOINTS  
            output["proj"] = weighted_mse_loss(projected_2d, self.projection_client.pose_2d_tensor, self.projection_scales, proj_N)

        #bone length consistency 
        if self.use_bone_term and not self.use_single_joint:
            if self.bone_len_method == "no_sqrt":
                bone_len_func = calculate_bone_lengths
            elif self.bone_len_method == "sqrt":
                bone_len_func = calculate_bone_lengths_sqrt
            
            if self.optimization_mode == "estimate_past":
                length_of_bone = bone_len_func(bones=self.pose3d[self.FUTURE_WINDOW_SIZE:], bone_connections=self.bone_connections, batch=True)
                bone_N = self.ESTIMATION_WINDOW_SIZE*(length_of_bone.shape[0])
                compare_bone_len = self.batch_bone_lengths[self.FUTURE_WINDOW_SIZE:]
            elif self.optimization_mode == "estimate_future":
                length_of_bone = bone_len_func(bones=self.pose3d[:self.FUTURE_WINDOW_SIZE], bone_connections=self.bone_connections, batch=True)
                bone_N = self.FUTURE_WINDOW_SIZE*(length_of_bone.shape[0])
                compare_bone_len = self.batch_bone_lengths[:self.FUTURE_WINDOW_SIZE,:]
            elif self.optimization_mode == "estimate_whole":
                length_of_bone = bone_len_func(bones=self.pose3d, bone_connections=self.bone_connections, batch=True)
                compare_bone_len = self.batch_bone_lengths
                bone_N = self.ONLINE_WINDOW_SIZE*(length_of_bone.shape[0])
            output["bone"] = mse_loss(length_of_bone, compare_bone_len, bone_N)
        
        #smoothness term
        if self.smoothness_mode == "velocity":
            if self.optimization_mode == "estimate_past":
                vel_tensor = self.pose3d[self.FUTURE_WINDOW_SIZE+1:, :, :] - self.pose3d[self.FUTURE_WINDOW_SIZE:-1, :, :]
                smooth_N = (self.ESTIMATION_WINDOW_SIZE)*self.NUM_OF_JOINTS
            elif self.optimization_mode == "estimate_whole" or self.optimization_mode == "estimate_future":
                vel_tensor = self.pose3d[1:, :, :] - self.pose3d[:-1, :, :]
                smooth_N = (self.ONLINE_WINDOW_SIZE)*self.NUM_OF_JOINTS
            output["smooth"] = mse_loss(vel_tensor[1:,:,:], vel_tensor[:-1,:,:], smooth_N)

        elif self.smoothness_mode == "position":
            raise NotImplementedError
            output["smooth"] = mse_loss(self.pose3d[1:, :, :], self.pose3d[:-1, :, :], (self.ONLINE_WINDOW_SIZE-1)*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == "all_connected":
            raise NotImplementedError
            vel_tensor = self.pose3d[:, :, :] - self.pose3d[self.n, :, :]
            output["smooth"] = mse_loss(vel_tensor[:,:,:], vel_tensor[self.n,:,:], self.ONLINE_WINDOW_SIZE*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == "only_velo_connected":
            raise NotImplementedError
            vel_tensor = self.pose3d[1:, :, :] - self.pose3d[:-1, :, :]
            output["smooth"] = mse_loss(vel_tensor[:,:,:], vel_tensor[self.m,:,:], (self.ONLINE_WINDOW_SIZE-1)*self.NUM_OF_JOINTS)

        #lift term  
        if not self.use_single_joint and self.use_lift_term:
            if self.lift_method == "complex":
                raise NotImplementedError
                if self.optimization_mode == "estimate_future":
                    output["lift"]=0
                elif self.optimization_mode == "estimate_past":
                    pose_est_directions = calculate_bone_directions(self.pose3d[self.FUTURE_WINDOW_SIZE:,:,:], self.lift_bone_directions, batch=True)
                    output["lift"] = mse_loss(self.pose3d_lift_directions, pose_est_directions,  self.ESTIMATION_WINDOW_SIZE*self.NUM_OF_JOINTS)
                elif self.optimization_mode == "estimate_whole":
                    pose_est_directions = calculate_bone_directions(self.pose3d, self.lift_bone_directions, batch=True)
                    output["lift"] = mse_loss(self.pose3d_lift_directions, pose_est_directions,  self.ONLINE_WINDOW_SIZE*self.NUM_OF_JOINTS)

            elif self.lift_method == "simple":
                if self.optimization_mode == "estimate_future":
                    output["lift"]=0
                elif self.optimization_mode == "estimate_past":
                    lift_N = self.ESTIMATION_WINDOW_SIZE*self.NUM_OF_JOINTS
                    output["lift"] = mse_loss(self.pose3d_lift_directions, self.pose3d[self.FUTURE_WINDOW_SIZE:, :, :]-self.pose3d[self.FUTURE_WINDOW_SIZE:, :, self.hip_index].unsqueeze(2), lift_N)
                elif self.optimization_mode == "estimate_whole":
                    lift_N = self.ONLINE_WINDOW_SIZE*self.NUM_OF_JOINTS
                    output["lift"] = mse_loss(self.pose3d_lift_directions, self.pose3d-self.pose3d[:, :, self.hip_index].unsqueeze(2), lift_N)

        overall_output = 0
        for loss_key in self.loss_dict:
            #print(loss_key)
            #$print(self.energy_weights[loss_key])
            #print(output[loss_key])
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])
            self.pltpts_weighted[loss_key].append(output[loss_key]*self.energy_weights[loss_key])
        return overall_output
    
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]



###################################################
################################################### 
###################################################

def project_trajectory(trajectory, window_size, number_of_traj_param):
    pose3d = torch.zeros(window_size, trajectory.shape[1], trajectory.shape[2])
    phase = (torch.arange(number_of_traj_param-1)%2).float() #change
    freq = (torch.arange(number_of_traj_param-1)//2+1).float()

    #form trajectory basis vectors
    cos_term = torch.zeros(window_size, number_of_traj_param-1)
    for t in range(window_size):
        cos_term[t, :] = torch.cos(freq*2*np.pi*t/(window_size-1) + phase*np.pi/2) 

    #orthonormalization using GS method.
    for param in range(number_of_traj_param-1):
        if param == 0:
            u_t = cos_term[:, param].clone()
        else:
            v_t = cos_term[:, param].clone()
            u_t = v_t - (torch.dot(v_t, u_t)/torch.dot(u_t, u_t))*u_t
        cos_term[:,param] = (u_t/torch.norm(u_t))
   
    for t in range(window_size):
        pose3d[t, :, :] = trajectory[0,:,:] + torch.sum(trajectory[1:,:,:]*cos_term[t, :].unsqueeze(1).unsqueeze(2).repeat(1, trajectory.shape[1], trajectory.shape[2]), dim=0)

    return pose3d

class pose3d_online_parallel_traj(torch.nn.Module):

    def __init__(self, pose_client, projection_client, lift_client, future_proj):
        super(pose3d_online_parallel_traj, self).__init__()
        self.future_proj = future_proj

        self.projection_client = projection_client
        self.num_of_traj_param = pose_client.NUMBER_OF_TRAJ_PARAM

        bone_connections, _, self.NUM_OF_JOINTS, _ = pose_client.model_settings()
        self.bone_connections = np.array(bone_connections)
        self.window_size = pose_client.ONLINE_WINDOW_SIZE

        self.pose3d = torch.nn.Parameter(torch.zeros(self.num_of_traj_param, 3, self.NUM_OF_JOINTS).to(pose_client.device), requires_grad=True)
        self.result_shape = pose_client.result_shape
        if pose_client.animation == "noise":
            self.bone_lengths = pose_client.multiple_bone_lengths
        else:
            self.bone_lengths = (pose_client.boneLengths).repeat(self.window_size,1)
        
        self.loss_dict = pose_client.loss_dict_online
        
        if self.future_proj:
            self.energy_weights = pose_client.weights_future
        else:
            self.energy_weights = pose_client.weights_online

        self.smoothness_mode = pose_client.SMOOTHNESS_MODE
        self.use_lift_term = pose_client.USE_LIFT_TERM
        self.use_bone_term = pose_client.USE_BONE_TERM
        self.use_single_joint = pose_client.USE_SINGLE_JOINT

        if self.use_lift_term and not self.use_single_joint:
            self.pose3d_lift_directions = lift_client.pose3d_lift_directions
            self.lift_bone_directions = np.array(return_lift_bone_connections(bone_connections))

        self.pltpts = {}
        self.pltpts_weighted = {}
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
            self.pltpts_weighted[loss_key] = []

        self.n = list(range(1, self.window_size))
        self.n.append(0)
        self.m = list(range(1, self.window_size))
        self.m.append(0)

    def forward(self):
        output = {}

        pose3d_projected = project_trajectory(self.pose3d, self.window_size, self.num_of_traj_param)

        #projection loss
        if not self.future_proj:
            projected_2d = self.projection_client.take_projection(pose3d_projected[1:,:,:])
        else:
            projected_2d = self.projection_client.take_projection(pose3d_projected)
        output["proj"] = mse_loss(projected_2d, self.projection_client.pose_2d_tensor, 2*self.NUM_OF_JOINTS)

        #smoothness term
        if self.smoothness_mode == "velocity":
            vel_tensor = pose3d_projected[1:, :, :] - pose3d_projected[:-1, :, :]
            output["smooth"] = mse_loss(vel_tensor[1:,:,:], vel_tensor[:-1,:,:], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == "position":
            output["smooth"] = mse_loss(pose3d_projected[1:, :, :], pose3d_projected[:-1, :, :], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == "all_connected":
            vel_tensor = pose3d_projected[:, :, :] - pose3d_projected[self.n, :, :]
            output["smooth"] = mse_loss(vel_tensor[:,:,:], vel_tensor[self.n,:,:], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == "only_velo_connected":
            vel_tensor = pose3d_projected[1:, :, :] - pose3d_projected[:-1, :, :]
            output["smooth"] = mse_loss(vel_tensor[:,:,:], vel_tensor[self.m,:,:], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == "none":
            output["smooth"] = 0

        if self.use_bone_term and not self.use_single_joint:
            #bone length consistency 
            length_of_bone = torch.sum(torch.pow(pose3d_projected[:, :, self.bone_connections[:,0]] - pose3d_projected[:, :, self.bone_connections[:,1]], 2), dim=1)
            bonelosses = torch.pow((length_of_bone - self.bone_lengths),2)
            output["bone"] = torch.sum(bonelosses)/(self.NUM_OF_JOINTS-1)

        #lift term  
        if not self.use_single_joint and self.use_lift_term:
            if not self.future_proj:
                pose_est_directions = calculate_bone_directions(pose3d_projected[1:,:,:], self.lift_bone_directions, batch=True)
            else:
                pose_est_directions = calculate_bone_directions(pose3d_projected, self.lift_bone_directions, batch=True)
            output["lift"] = mse_loss(self.pose3d_lift_directions, pose_est_directions,  3*self.NUM_OF_JOINTS)


        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])

        return overall_output

    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]