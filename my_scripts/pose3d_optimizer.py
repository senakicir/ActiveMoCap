import torch
import torch.nn as nn
from helpers import * 
from project_bones import take_bone_projection_pytorch
from torch.autograd import Variable

def mse_loss(input, target):
    return torch.sum(torch.pow((input - target),2)) / input.data.nelement()

class pose3d_calibration(torch.nn.Module):

    def __init__(self, model):
        super(pose3d_calibration, self).__init__()
        self.bone_connections, _, self.NUM_OF_JOINTS, _ = model_settings(model)


        self.pose3d = torch.nn.Parameter(torch.zeros([3, self.NUM_OF_JOINTS]), requires_grad=True)

    def forward(self, pose_2d, R_drone, C_drone):
        outputs = {}
        for loss in CALIBRATION_LOSSES:
            outputs[loss] = 0

        projected_2d, _ = take_bone_projection_pytorch(self.pose3d, R_drone, C_drone)

        outputs["proj"] = mse_loss(projected_2d, pose_2d)

        left_bone_connections, right_bone_connections, _ = split_bone_connections(self.bone_connections)
        bonelosses = Variable(torch.zeros([len(left_bone_connections),1]), requires_grad = False)
        for i, l_bone in enumerate(left_bone_connections):
            r_bone = right_bone_connections[i]
            left_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, l_bone[0]] - self.pose3d[:, l_bone[1]], 2)))
            right_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, r_bone[0]] - self.pose3d[:, r_bone[1]], 2)))
            bonelosses[i] = torch.pow((left_length_of_bone - right_length_of_bone),2)
        outputs["sym"] = torch.sum(bonelosses)/bonelosses.data.nelement()
        return outputs
    
    def init_pose3d(self, pose3d_):
        self.pose3d.data[:] = pose3d_.data[:]


class pose3d_flight(torch.nn.Module):

    def __init__(self, bone_lengths_, window_size_, model):
        super(pose3d_flight, self).__init__()
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.window_size = window_size_
        self.pose3d = torch.nn.Parameter(torch.zeros([self.window_size, 3, self.NUM_OF_JOINTS]), requires_grad=True)
        self.bone_lengths = Variable(bone_lengths_, requires_grad = False)

    def forward(self, pose_2d, R_drone, C_drone, pose3d_lift_directions, queue_index):
        #projection loss
        outputs = {}
        for loss in LOSSES:
            outputs[loss] = 0
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d[queue_index, :, :].cpu(), R_drone, C_drone)
        outputs["proj"] = mse_loss(projected_2d, pose_2d)

        #smoothness
        #if (queue_index != self.window_size-1 and queue_index != 0):
        #    outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :] - self.pose3d[queue_index+1, :, :], self.pose3d[queue_index-1, :, :]- self.pose3d[queue_index, :, :])
        if (queue_index != self.window_size-1 and queue_index != 0):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :]) +  mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])
        elif (queue_index != self.window_size-1 ):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :])
        elif (queue_index != 0):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])

        #bone length consistency 
        bonelosses = Variable(torch.zeros([self.NUM_OF_JOINTS-1,1]), requires_grad = False)
        for i, bone in enumerate(self.bone_connections):
            length_of_bone = (torch.sum(torch.pow(self.pose3d[queue_index, :, bone[0]] - self.pose3d[queue_index, :, bone[1]], 2)))
            bonelosses[i] = torch.pow((self.bone_lengths[i] - length_of_bone),2)
        outputs["bone"] = torch.sum(bonelosses)/bonelosses.data.nelement()

        pose_est_directions = torch.zeros([3, self.NUM_OF_JOINTS-1])
        for i, bone in enumerate(self.bone_connections):
            bone_vector = self.pose3d[queue_index, :, bone[0]] - self.pose3d[queue_index, :, bone[1]]
            pose_est_directions[:, i] = bone_vector/(torch.norm(bone_vector)+EPSILON)

        #hip_index = self.joint_names.index('spine1')
        #hip = self.pose3d[queue_index, :, hip_index].unsqueeze(1)
        #temp_pose3d_t = torch.sub(self.pose3d[queue_index, :, :], hip)
        #normalized_pose_3d, temp_pose3d_t = normalize_pose(self.pose3d[queue_index, :, :], self.joint_names, is_torch = True)


        #if (queue_index != self.window_size-1 and queue_index != 0):
        #    temp_pose3d_t_p_1 = torch.sub(self.pose3d[queue_index+1, :, :], self.pose3d[queue_index+1, :, hip_index].unsqueeze(1))
        #    temp_pose3d_t_m_1 = torch.sub(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index-1, :, hip_index].unsqueeze(1))
        #    outputs["smoothpose"] = mse_loss(temp_pose3d_t, temp_pose3d_t_p_1) +  mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)
        #elif (queue_index != self.window_size-1 ):
        #    temp_pose3d_t_p_1 = torch.sub(self.pose3d[queue_index+1, :, :], self.pose3d[queue_index+1, :, hip_index].unsqueeze(1))
        #    outputs["smoothpose"] = mse_loss(temp_pose3d_t, temp_pose3d_t_p_1)
        #elif (queue_index != 0):
        #    temp_pose3d_t_m_1 = torch.sub(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index-1, :, hip_index].unsqueeze(1))
        #    outputs["smoothpose"] = mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)

        #normalize pose
        #max_z = torch.max(temp_pose3d_t[2,:])
        #min_z = torch.min(temp_pose3d_t[2,:])
        #normalized_pose_3d = (pose3d_lift)/(max_z - min_z)
        outputs["lift"]= mse_loss(pose_est_directions, pose3d_lift_directions)

        return outputs
    
    def init_pose3d(self, pose3d_, queue_index):
        self.pose3d.data[queue_index, :, :] = pose3d_.data[:]


