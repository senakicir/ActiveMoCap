import torch as torch
from helpers import EPSILON
from PoseEstimationClient import calculate_bone_lengths, calculate_bone_lengths_sqrt
import pdb

def calculate_bone_directions(bones, lift_bone_directions, batch):
    if batch:
        current_bone_vector = bones[:, :, lift_bone_directions[:,0]] - bones[:, :, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=1, keepdim=True)).repeat(1,3,1) #try without repeat
    else:
        current_bone_vector = bones[:, lift_bone_directions[:,0]] - bones[:, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=0, keepdim=True)).repeat(3,1) #try without repeat
    return (current_bone_vector/(norm_bone_vector+EPSILON)).float()

def calculate_bone_directions_simple(lift_bones, bone_lengths, bone_length_method, bone_connections, hip_index, batch):
    if bone_length_method == "no_sqrt":
        calculate_bone_lengths_func = calculate_bone_lengths
    elif bone_length_method == "sqrt":
        calculate_bone_lengths_func = calculate_bone_lengths_sqrt

    lift_bone_lengths = calculate_bone_lengths_func(lift_bones, bone_connections, batch)
    if batch:
        scale = torch.sum(bone_lengths.unsqueeze(0) * lift_bone_lengths, dim=1)/torch.sum(lift_bone_lengths**2, dim=1)
        lift_bone_rescaled  = scale[:, None, None]*lift_bones
        lift_bone_rescaled = lift_bone_rescaled - lift_bone_rescaled[:, :, hip_index].unsqueeze(2)
    else:
        scale = torch.sum(bone_lengths * lift_bone_lengths)/torch.sum(lift_bone_lengths**2)
        lift_bone_rescaled = scale*lift_bones
        lift_bone_rescaled = lift_bone_rescaled - lift_bone_rescaled[:, hip_index].unsqueeze(1)
    return lift_bone_rescaled

class Lift_Client(object):
    def reset(self, lift_list):
        self.pose3d_lift_directions = (torch.stack(lift_list.copy()))

    def reset_future(self, lift_list, potential_lift_directions):
        temp = torch.stack(lift_list.copy())
        self.pose3d_lift_directions = torch.cat((potential_lift_directions.unsqueeze(0), temp), dim=0)

        
