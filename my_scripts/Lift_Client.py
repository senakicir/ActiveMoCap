import torch as torch
from helpers import EPSILON

def calculate_bone_directions(bones, lift_bone_directions, batch):
    if batch:
        current_bone_vector = bones[:, :, lift_bone_directions[:,0]] - bones[:, :, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=1, keepdim=True)).repeat(1,3,1) #try without repeat
    else:
        current_bone_vector = bones[:, lift_bone_directions[:,0]] - bones[:, lift_bone_directions[:,1]]
        norm_bone_vector = (torch.norm(current_bone_vector, dim=0, keepdim=True)).repeat(3,1) #try without repeat
    return (current_bone_vector/(norm_bone_vector+EPSILON)).float()

class Lift_Client(object):
    def reset(self, lift_list):
        self.pose3d_lift_directions = (torch.stack(lift_list.copy()))

    def reset_future(self, lift_list, potential_lift_directions):
        temp = torch.stack(lift_list.copy())
        self.pose3d_lift_directions = torch.cat((potential_lift_directions.unsqueeze(0), temp), dim=0)

        
