import torch as torch
from helpers import EPSILON
from pose_helper_functions import calculate_bone_lengths, calculate_bone_lengths_sqrt, add_noise_to_pose
import pdb

def scale_with_bone_lengths(pose_to_scale, bone_lengths, bone_length_method, bone_connections):
    if bone_length_method == "no_sqrt":
        calculate_bone_lengths_func = calculate_bone_lengths
    elif bone_length_method == "sqrt":
        calculate_bone_lengths_func = calculate_bone_lengths_sqrt
    our_pose_bone_lengths = calculate_bone_lengths_func(pose_to_scale, bone_connections, batch=False)
    scale = torch.sum(bone_lengths * our_pose_bone_lengths)/torch.sum(our_pose_bone_lengths**2)
    return scale*pose_to_scale

def calculate_bone_directions(bones, lift_bone_directions, batch):
    if batch:
        current_bone_vector = bones[:, :, lift_bone_directions[:,0]] - bones[:, :, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=1, keepdim=True)).repeat(1,3,1) #try without repeat
    else:
        current_bone_vector = bones[:, lift_bone_directions[:,0]] - bones[:, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=0, keepdim=True)).repeat(3,1) #try without repeat
    return (current_bone_vector/(norm_bone_vector+EPSILON)).float()

def calculate_bone_directions_simple(lift_bones, bone_lengths, bone_length_method, bone_connections, hip_index):
    lift_bone_rescaled = scale_with_bone_lengths(lift_bones, bone_lengths, bone_length_method, bone_connections)
    lift_bone_rescaled -= lift_bone_rescaled[:, hip_index].unsqueeze(1)
    return lift_bone_rescaled

class Lift_Client(object):
    def reset(self, lift_list, bone_3d_pose_gt, simulate_error_mode, noise_lift_std):
        if simulate_error_mode:
            temp = torch.from_numpy(bone_3d_pose_gt.copy()).float()
            temp = add_noise_to_pose(temp, noise_lift_std)
        else:
            temp = torch.stack(lift_list.copy()).float()

        self.pose3d_lift_directions = temp

    def reset_future(self, lift_list, potential_lift_directions, simulate_error_mode, noise_lift_std):
        temp = torch.stack(lift_list.copy()).float()
        self.pose3d_lift_directions = torch.cat((potential_lift_directions.unsqueeze(0), temp), dim=0)

        
