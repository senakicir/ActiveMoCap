import torch as torch
from math import pi, cos, sin, degrees
import numpy as np
from helpers import add_noise_to_pose, euler_to_rotation_matrix, do_nothing, CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z, CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET, px, py, FOCAL_LENGTH, SIZE_X, SIZE_Y 
import pdb

neat_tensor = torch.FloatTensor([[0, 0, 0, 1]]) #this tensor is neat!
DEFAULT_TORSO_SIZE = 0.125

C_cam_torch = torch.FloatTensor([[CAMERA_OFFSET_X], [CAMERA_OFFSET_Y], [CAMERA_OFFSET_Z]])

FLIP_X_Y_torch = torch.FloatTensor([[0,1,0],[-1,0,0],[0,0,1]])

FLIP_X_Y_inv_torch = torch.inverse(FLIP_X_Y_torch)

class Projection_Client_Old(object):
    def reset(self, data_list, num_of_joints, simulate_error_mode, noise_2d_std):
        self.num_of_joints = num_of_joints
        self.window_size = len(data_list)
        self.drone_transformation = torch.zeros(self.window_size , 4, 4)
        self.camera_transformation = torch.zeros(self.window_size , 4, 4)

        queue_index = 0
        self.pose_2d_tensor = torch.zeros(self.window_size , 2, num_of_joints)
    
        for bone_2d, bone_2d_gt, R_drone_torch, C_drone_torch, R_cam_torch in data_list:
            if simulate_error_mode:
                bone_2d = add_noise_to_pose(bone_2d_gt.float(), noise_2d_std)
            self.pose_2d_tensor[queue_index, :, :] = bone_2d.clone()
            self.drone_transformation[queue_index, :, :]= torch.inverse(torch.cat((torch.cat((R_drone_torch, C_drone_torch), dim=1), neat_tensor), dim=0) )
            self.camera_transformation[queue_index, :, :]= torch.inverse(torch.cat((torch.cat((R_cam_torch, C_cam_torch), dim=1), neat_tensor), dim=0) )

            queue_index += 1

        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints)
        self.flip_x_y_tensor = (torch.cat((FLIP_X_Y_torch, torch.zeros(3,1)), dim=1)).repeat(self.window_size , 1, 1)
        self.camera_intrinsics = K_torch.repeat(self.window_size , 1,1)

    def reset_future(self, data_list, num_of_joints, R_cam_pot, R_drone_pot, C_drone_pot, potential_projected_est):
        self.num_of_joints = num_of_joints
        self.window_size = len(data_list)+1
        self.drone_transformation = torch.zeros(self.window_size , 4, 4)
        self.camera_transformation = torch.zeros(self.window_size , 4, 4)
        self.pose_2d_tensor = torch.zeros(self.window_size, 2, num_of_joints)

        self.pose_2d_tensor[0, :, :] = potential_projected_est
        self.drone_transformation[0, :, :]= torch.inverse(torch.cat((torch.cat((R_drone_pot, C_drone_pot), dim=1), neat_tensor), dim=0))
        self.camera_transformation[0, :, :]= torch.inverse(torch.cat((torch.cat((R_cam_pot, C_cam_torch), dim=1), neat_tensor), dim=0) )

        queue_index = 1
        for bone_2d, _, R_drone_torch, C_drone_torch, R_cam_torch in data_list:
            self.pose_2d_tensor[queue_index, :, :] = bone_2d.clone()
            self.drone_transformation[queue_index, :, :]= torch.inverse(torch.cat((torch.cat((R_drone_torch, C_drone_torch), dim=1), neat_tensor), dim=0) )
            self.camera_transformation[queue_index, :, :]= torch.inverse(torch.cat((torch.cat((R_cam_torch, C_cam_torch), dim=1), neat_tensor), dim=0) )
            queue_index += 1

        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints)
        self.flip_x_y_tensor = (torch.cat((FLIP_X_Y_torch, torch.zeros(3,1)), dim=1)).repeat(self.window_size , 1, 1)
        self.camera_intrinsics = K_torch.repeat(self.window_size , 1,1)

    def take_projection(self, pose_3d):
        P_world = torch.cat((pose_3d, self.ones_tensor), dim=1)
        P_drone = torch.bmm(self.drone_transformation, P_world)
        P_camera_temp = torch.bmm(self.camera_transformation, P_drone)

        P_camera = torch.bmm(self.flip_x_y_tensor , P_camera_temp)
        proj_homog = torch.bmm(self.camera_intrinsics , P_camera)

        z = proj_homog[:,2,:]
        result = torch.zeros(self.window_size, 2, self.num_of_joints)
        result[:,0,:] = proj_homog[:,0,:]/z
        result[:,1,:] = proj_homog[:,1,:]/z
        return result


class Projection_Client(object):
    def __init__(self, focal_length=FOCAL_LENGTH, px=px, py=py):
        self.K_torch = (torch.FloatTensor([[focal_length,0,px],[0,focal_length,py],[0,0,1]]))
        self.K_inv_torch = torch.inverse(self.K_torch)
        self.focal_length = focal_length

    def reset(self, data_list, num_of_joints, simulate_error_mode, noise_2d_std):
        self.num_of_joints = num_of_joints
        self.window_size = len(data_list)
        self.transformation_matrix = torch.zeros(self.window_size , 4, 4)

        queue_index = 0
        self.pose_2d_tensor = torch.zeros(self.window_size , 2, num_of_joints)
    
        for bone_2d, bone_2d_gt, transformation_matrix in data_list:
            if simulate_error_mode:
                bone_2d = add_noise_to_pose(bone_2d_gt.float(), noise_2d_std)
            self.pose_2d_tensor[queue_index, :, :] = bone_2d.clone()
            self.transformation_matrix[queue_index, :, :]= transformation_matrix.clone()
            queue_index += 1

        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints)
        self.flip_x_y_tensor = (torch.cat((FLIP_X_Y_torch, torch.zeros(3,1)), dim=1)).repeat(self.window_size , 1, 1)
        self.camera_intrinsics = self.K_torch.repeat(self.window_size , 1,1)

    def reset_future(self, data_list, num_of_joints, R_cam_pot, R_drone_pot, C_drone_pot, potential_projected_est):
        self.num_of_joints = num_of_joints
        self.window_size = len(data_list)+1
        self.drone_transformation = torch.zeros(self.window_size , 4, 4)
        self.camera_transformation = torch.zeros(self.window_size , 4, 4)
        self.pose_2d_tensor = torch.zeros(self.window_size, 2, num_of_joints)

        self.pose_2d_tensor[0, :, :] = potential_projected_est
        self.drone_transformation[0, :, :]= torch.inverse(torch.cat((torch.cat((R_drone_pot, C_drone_pot), dim=1), neat_tensor), dim=0))
        self.camera_transformation[0, :, :]= torch.inverse(torch.cat((torch.cat((R_cam_pot, C_cam_torch), dim=1), neat_tensor), dim=0) )

        queue_index = 1
        for bone_2d, _, inverse_transformation_matrix in data_list:
            self.pose_2d_tensor[queue_index, :, :] = bone_2d.clone()
            self.inverse_transformation_matrix[queue_index, :, :]= inverse_transformation_matrix.clone()
            queue_index += 1

        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints)
        self.flip_x_y_tensor = (torch.cat((FLIP_X_Y_torch, torch.zeros(3,1)), dim=1)).repeat(self.window_size , 1, 1)
        self.camera_intrinsics = self.K_torch.repeat(self.window_size , 1,1)

    def take_projection(self, pose_3d):
        P_world = torch.cat((pose_3d, self.ones_tensor), dim=1)
        self.world_to_camera(P_world, inverse_transformation_matrix)
  
        proj_homog = torch.bmm(self.camera_intrinsics , P_camera)

        z = proj_homog[:,2,:]
        result = torch.zeros(self.window_size, 2, self.num_of_joints)
        result[:,0,:] = proj_homog[:,0,:]/z
        result[:,1,:] = proj_homog[:,1,:]/z
        return result

    def world_to_camera(self, P_world, inv_transformation_matrix)
        P_camera = torch.bmm(inverse_transformation_matrix, P_world)
        P_camera = P_camera[0:3,:]
        P_camera = torch.mm(self.flip_x_y_tensor, P_camera)
        return P_camera    

    def take_single_projection(self, P_world, inv_transformation_matrix):
        ones_tensor = torch.ones([1, self.num_of_joints])*1.0
        P_world = torch.cat((P_world, ones_tensor), 0)

        P_camera = world_to_camera(P_world, inv_transformation_matrix)
        x = torch.mm(self.K_torch, P_camera)
        
        z = x[2,:]
        result = torch.zeros([2, self.num_of_joints])
        result[0,:] = x[0,:]/z
        result[1,:] = x[1,:]/z
        return result

    def take_single_backprojection(self, pose_2d, transformation_matrix, joint_names):
        ones_tensor = torch.ones([1, self.num_of_joints])*1.0
        img_torso_size = torch.norm(pose_2d[:, joint_names.index('neck')] - pose_2d[:, joint_names.index('spine1')])
        z_val = ((SELF.focal_length * DEFAULT_TORSO_SIZE) / img_torso_size)

        bone_pos_3d = torch.zeros([3, self.num_of_joints])
        bone_pos_3d[0,:] = pose_2d[0,:]*z_val
        bone_pos_3d[1,:] = pose_2d[1,:]*z_val
        bone_pos_3d[2,:] = ones_tensor*z_val
        
        bone_pos_3d = torch.mm(self.K_inv_torch, bone_pos_3d)
        P_world = camera_to_world(bone_pos_3d, transformation_matrix)

        return P_world

    def camera_to_world(self, P_camera, transformation_matrix):
        P_camera = torch.mm(FLIP_X_Y_inv_torch, P_camera)
        ones_tensor = torch.ones([1, self.num_of_joints])*1.0
        P_camera = torch.cat((P_camera, ones_tensor),0)

        P_world_ = torch.bmm(transformation_matrix, P_camera)
        P_world = P_world_.clone()
        return P_world    

    def take_potential_projection(self, potential_state, future_pose):
        C_drone = potential_state["position"].copy()
        potential_C_drone = torch.from_numpy(C_drone[:, np.newaxis]).float()
        yaw = potential_state["orientation"]
        potential_R_drone = euler_to_rotation_matrix(0,0,yaw, returnTensor=True)
        camera_pitch = potential_state["pitch"]
        potential_R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, camera_pitch+pi/2, CAMERA_YAW_OFFSET, returnTensor=True)
        potential_future_pose = torch.from_numpy(future_pose.copy()).float()

        drone_transformation = torch.cat((torch.cat((potential_R_drone, potential_C_drone), dim=1), neat_tensor), dim=0)
        camera_transformation = torch.cat((torch.cat((potential_R_cam, C_cam_torch), dim=1), neat_tensor), dim=0) 
        inv_drone_transformation_matrix = torch.inverse(drone_transformation@camera_transformation)

        proj =  take_single_projection(potential_future_pose, inv_drone_transformation_matrix)
        return proj.numpy()