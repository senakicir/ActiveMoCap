import torch

def add_noise_to_pose(pose, noise_std):
    noise = torch.normal(torch.zeros(pose.shape), torch.ones(pose.shape)*noise_std).float()
    if (pose.is_cuda):
        my_device = torch.device("cuda")
        noise = noise.to(my_device)
    pose = pose.float().clone() + noise
    return pose 


def calculate_bone_lengths(bones, bone_connections, batch):
    if batch:
        return (torch.sum(torch.pow(bones[:, :, bone_connections[:,0]] - bones[:, :, bone_connections[:,1]], 2), dim=1)).float()
    else:  
        return (torch.sum(torch.pow(bones[:, bone_connections[:,0]] - bones[:, bone_connections[:,1]], 2), dim=0)).float()

def calculate_bone_lengths_sqrt(bones, bone_connections, batch):
    if batch:
        return torch.sqrt(torch.sum(torch.pow(bones[:, :, bone_connections[:,0]] - bones[:, :, bone_connections[:,1]], 2), dim=1)).float()
    else:  
        return torch.sqrt(torch.sum(torch.pow(bones[:, bone_connections[:,0]] - bones[:, bone_connections[:,1]], 2), dim=0)).float()    
