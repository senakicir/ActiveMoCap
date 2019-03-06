from project_bones import take_bone_projection_pytorch, Projection_Client
import torch as torch
from helpers import model_settings, split_bone_connections, EPSILON, return_lift_bone_connections, euler_to_rotation_matrix, CAMERA_PITCH_OFFSET, CAMERA_ROLL_OFFSET, CAMERA_YAW_OFFSET, CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z
import numpy as np 
from math import pi
from PoseEstimationClient import calculate_bone_lengths, calculate_bone_directions

def mse_loss(input_1, input_2, N):
    return torch.sum(torch.pow((input_1 - input_2),2))/N

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
        self.pose3d = torch.nn.Parameter(torch.zeros([4,1]), requires_grad=True)

    def forward(self):
        return self.pose3d[3]**2 + self.pose3d[2]*self.pose3d[1]*self.pose3d[0] + self.pose3d[0]**4
 
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_calibration_parallel(torch.nn.Module):

    def __init__(self, pose_client, projection_client):
        super(pose3d_calibration_parallel, self).__init__()
        self.bone_connections, _, self.NUM_OF_JOINTS = model_settings(pose_client.model)
        left_bone_connections, right_bone_connections, _ = split_bone_connections(self.bone_connections)
        self.left_bone_connections = np.array(left_bone_connections)
        self.right_bone_connections = np.array(right_bone_connections)
        self.pose3d = torch.nn.Parameter(torch.zeros(pose_client.result_shape), requires_grad=True)

        self.energy_weights = pose_client.weights_calib
        self.loss_dict = pose_client.loss_dict_calib
        self.projection_client = projection_client
        self.use_symmetry_term = pose_client.USE_SYMMETRY_TERM

        self.pltpts = {}
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []

    def forward(self):        
        output = {}
        if self.use_symmetry_term :
            left_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, self.left_bone_connections[:,0]] - self.pose3d[:, self.left_bone_connections[:,1]], 2), dim=0))
            right_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, self.right_bone_connections[:,0]] - self.pose3d[:, self.right_bone_connections[:,1]], 2), dim=0))
            bonelosses = torch.pow((left_length_of_bone - right_length_of_bone),2)
            output["sym"] = torch.sum(bonelosses)/6

        rep_pose3d = self.pose3d.repeat(self.projection_client.window_size, 1, 1)
        projected_2d = self.projection_client.take_projection(rep_pose3d)
        output["proj"] = mse_loss(projected_2d, self.projection_client.pose_2d_tensor, 2*self.NUM_OF_JOINTS)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])

        return overall_output

    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_online_parallel(torch.nn.Module):

    def __init__(self, pose_client, projection_client):
        super(pose3d_online_parallel, self).__init__()

        self.projection_client = projection_client

        bone_connections, self.joint_names, self.NUM_OF_JOINTS = model_settings(pose_client.model)
        self.bone_connections = np.array(bone_connections)
        self.window_size = pose_client.ONLINE_WINDOW_SIZE

        self.pose3d = torch.nn.Parameter(torch.zeros(pose_client.result_shape), requires_grad=True)

        self.bone_lengths = (pose_client.boneLengths).repeat(self.window_size+1,1)
        self.loss_dict = pose_client.loss_dict_online
        
        self.pose3d_lift_directions = torch.stack(pose_client.liftPoseList).float()

        self.energy_weights = pose_client.weights_online
        self.lift_bone_directions = np.array(return_lift_bone_connections(bone_connections))

        self.smoothness_mode = pose_client.SMOOTHNESS_MODE
        self.use_lift_term = pose_client.USE_LIFT_TERM

        self.pltpts = {}
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []

        self.n = list(range(1, self.window_size+1))
        self.n.append(0)
        self.m = list(range(1, self.window_size))
        self.m.append(0)


    def forward(self):
        output = {}

        #projection loss
        projected_2d = self.projection_client.take_projection(self.pose3d[1:,:,:])
        output["proj"] = mse_loss(projected_2d, self.projection_client.pose_2d_tensor, 2*self.NUM_OF_JOINTS)

        #bone length consistency 
        length_of_bone = torch.sum(torch.pow(self.pose3d[:, :, self.bone_connections[:,0]] - self.pose3d[:, :, self.bone_connections[:,1]], 2), dim=1)
        bonelosses = torch.pow((length_of_bone - self.bone_lengths),2)
        output["bone"] = torch.sum(bonelosses)/(self.NUM_OF_JOINTS-1)

        #smoothness term
        if self.smoothness_mode == 0:
            vel_tensor = self.pose3d[1:, :, :] - self.pose3d[:-1, :, :]
            output["smooth"] = mse_loss(vel_tensor[1:,:,:], vel_tensor[:-1,:,:], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == 1:
            output["smooth"] = mse_loss(self.pose3d[1:, :, :], self.pose3d[:-1, :, :], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == 2:
            vel_tensor = self.pose3d[:, :, :] - self.pose3d[self.n, :, :]
            output["smooth"] = mse_loss(vel_tensor[:,:,:], vel_tensor[self.n,:,:], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == 3:
            vel_tensor = self.pose3d[1:, :, :] - self.pose3d[:-1, :, :]
            output["smooth"] = mse_loss(vel_tensor[:,:,:], vel_tensor[self.m,:,:], 3*self.NUM_OF_JOINTS)

        #lift term        
        if self.use_lift_term:
            pose_est_directions = calculate_bone_directions(self.pose3d, self.lift_bone_directions, batch=True)
            output["lift"] = mse_loss(self.pose3d_lift_directions, pose_est_directions,  3*self.NUM_OF_JOINTS)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])

        return overall_output
    
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_future_parallel(torch.nn.Module):

    def __init__(self, pose_client, projection_client):
        super(pose3d_future_parallel, self).__init__()

        self.projection_client = projection_client

        bone_connections, self.joint_names, self.NUM_OF_JOINTS = model_settings(pose_client.model)
        self.bone_connections = np.array(bone_connections)
        self.window_size = pose_client.ONLINE_WINDOW_SIZE

        self.pose3d = torch.nn.Parameter(torch.zeros(pose_client.result_shape), requires_grad=True)

        self.bone_lengths = (pose_client.boneLengths).repeat(self.window_size+1,1)
        self.loss_dict = pose_client.loss_dict_future
        
        self.pose3d_lift_directions = torch.stack(pose_client.liftPoseList).float()

        self.smoothness_mode  = pose_client.SMOOTHNESS_MODE
        self.use_lift_term = pose_client.USE_LIFT_TERM

        self.energy_weights = pose_client.weights_future
        self.lift_bone_directions = np.array(return_lift_bone_connections(bone_connections))
        self.pltpts = {}

        self.n = list(range(1, self.window_size+1))
        self.n.append(0)
        self.m = list(range(1, self.window_size))
        self.m.append(0)

        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
       

    def forward(self):
        output = {}

        #projection loss
        projected_2d = self.projection_client.take_projection(self.pose3d)
        output["proj"] =  mse_loss(projected_2d, self.projection_client.pose_2d_tensor, 2*self.NUM_OF_JOINTS)

        #bone length consistency 
        length_of_bone = calculate_bone_lengths(bones=self.pose3d, bone_connections=self.bone_connections, batch=True)
        bonelosses = torch.pow((length_of_bone - self.bone_lengths),2)
        output["bone"] = torch.sum(bonelosses)/(self.NUM_OF_JOINTS-1)

        #smoothness term
        if self.smoothness_mode == 0:
            vel_tensor = self.pose3d[1:, :, :] - self.pose3d[:-1, :, :]
            output["smooth"] = mse_loss(vel_tensor[1:,:,:], vel_tensor[:-1,:,:], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == 1:
            output["smooth"] = mse_loss(self.pose3d[1:,:,:], self.pose3d[:-1,:,:], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == 2:
            vel_tensor = self.pose3d[:, :, :] - self.pose3d[self.n, :, :]
            output["smooth"] = mse_loss(vel_tensor[:,:,:], vel_tensor[self.n,:,:], 3*self.NUM_OF_JOINTS)
        elif self.smoothness_mode == 3:
            vel_tensor = self.pose3d[1:, :, :] - self.pose3d[:-1, :, :]
            output["smooth"] = mse_loss(vel_tensor[:,:,:], vel_tensor[self.m,:,:], 3*self.NUM_OF_JOINTS)

        #lift term        
        if self.use_lift_term:
            pose_est_directions = calculate_bone_directions(self.pose3d, self.lift_bone_directions, batch=True)
            output["lift"] = mse_loss(self.pose3d_lift_directions, pose_est_directions,  3*self.NUM_OF_JOINTS)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])
        return overall_output
    
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]