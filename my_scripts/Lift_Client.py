import torch as torch
from helpers import EPSILON
from pose_helper_functions import calculate_bone_lengths, calculate_bone_lengths_sqrt, add_noise_to_pose


def scale_with_bone_lengths(pose_to_scale, bone_lengths, bone_length_method, bone_connections, batch):
    if bone_length_method == "no_sqrt":
        calculate_bone_lengths_func = calculate_bone_lengths
    elif bone_length_method == "sqrt":
        calculate_bone_lengths_func = calculate_bone_lengths_sqrt
    our_pose_bone_lengths = calculate_bone_lengths_func(pose_to_scale, bone_connections, batch=batch)

    if batch:
        scale = torch.sum(bone_lengths)/torch.sum(our_pose_bone_lengths, dim=1)
        scale = scale.unsqueeze(1).unsqueeze(1)
    else:
        scale = torch.sum(bone_lengths)/torch.sum(our_pose_bone_lengths)

    if bone_length_method == "no_sqrt":
        return torch.sqrt(scale)*pose_to_scale
    elif bone_length_method == "sqrt":
        return scale*pose_to_scale

def calculate_bone_directions(bones, lift_bone_directions, batch):
    if batch:
        current_bone_vector = bones[:, :, lift_bone_directions[:,0]] - bones[:, :, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=1, keepdim=True)).repeat(1,3,1) #try without repeat
    else:
        current_bone_vector = bones[:, lift_bone_directions[:,0]] - bones[:, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=0, keepdim=True)).repeat(3,1) #try without repeat
    return (current_bone_vector/(norm_bone_vector+EPSILON)).float()

def calculate_bone_directions_simple(lift_bones, bone_lengths, bone_length_method, bone_connections, hip_index, batch):
    lift_bone_rescaled = scale_with_bone_lengths(lift_bones, bone_lengths, bone_length_method, bone_connections, batch)
    if batch:
        lift_bone_rescaled = lift_bone_rescaled - lift_bone_rescaled[:, :, hip_index].unsqueeze(2)
    else:
        lift_bone_rescaled = lift_bone_rescaled - lift_bone_rescaled[:, hip_index].unsqueeze(1)
    return lift_bone_rescaled

class Lift_Client(object):
    def __init__(self, noise_lift_std, estimation_window_size, future_window_size):
        self.noise_lift_std = noise_lift_std
        self.estimation_window_size = estimation_window_size
        self.future_window_size = future_window_size

    def deepcopy_lift_client(self):
        return Lift_Client(self.noise_lift_std, self.estimation_window_size, self.future_window_size)

    def reset(self, lift_pose_tensor, bone_3d_pose_gt):
        self.pose3d_lift_directions = lift_pose_tensor.clone()

    def reset_future(self, lift_pose_tensor, potential_lift_directions, use_hessian_mode):
        temp = lift_pose_tensor.clone()
        if use_hessian_mode == "partial":
            temp = lift_pose_tensor[:(self.estimation_window_size-self.future_window_size)].clone()
        self.pose3d_lift_directions = torch.cat((potential_lift_directions, temp), dim=0)
