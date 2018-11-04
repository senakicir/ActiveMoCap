import torch
from torch.autograd import Variable
from math import pi, cos, sin, degrees
import numpy as np
from helpers import euler_to_rotation_matrix, do_nothing, CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z, CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET, CAMERA_YAW_OFFSET, px, py, FOCAL_LENGTH, SIZE_X, SIZE_Y 
import pdb

neat_tensor = Variable(torch.FloatTensor([[0, 0, 0, 1]]), requires_grad=False) #this tensor is neat!
ones_tensor = Variable(torch.ones([1, 15]), requires_grad=False)*1.0
DEFAULT_TORSO_SIZE = 0.42 #0.86710678118

#R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET+pi/2, CAMERA_YAW_OFFSET, returnTensor = False)
C_cam = np.array([[CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z]]).T
#R_cam_torch = Variable(torch.from_numpy(R_cam).float(), requires_grad = False)
C_cam_torch = Variable(torch.FloatTensor([[CAMERA_OFFSET_X], [CAMERA_OFFSET_Y], [CAMERA_OFFSET_Z]]), requires_grad = False)

FLIP_X_Y = np.array([[0,1,0],[-1,0,0],[0,0,1]])
FLIP_X_Y_torch = Variable(torch.FloatTensor([[0,1,0],[-1,0,0],[0,0,1]]), requires_grad = False)
UNREAL_MAPPING_FUNC = Variable(torch.FloatTensor([[0,1,0,0],[-1,0,0,0],[0,0,1,0]]), requires_grad = False)


FLIP_X_Y_inv = np.linalg.inv(FLIP_X_Y)
FLIP_X_Y_inv_torch = torch.inverse(FLIP_X_Y_torch)

K = np.array([[FOCAL_LENGTH,0,px],[0,FOCAL_LENGTH,py],[0,0,1]])
K_torch = Variable(torch.FloatTensor([[FOCAL_LENGTH,0,px],[0,FOCAL_LENGTH,py],[0,0,1]]), requires_grad = False)
K_inv = np.linalg.inv(K)
K_inv_torch = torch.inverse(K_torch)

def update_torso_size(new_size):
    #global DEFAULT_TORSO_SIZE
    #DEFAULT_TORSO_SIZE = new_size.numpy()[0]
    do_nothing

def take_bone_projection(P_world, R_drone, C_drone, R_cam):
    P_camera = world_to_camera(R_drone, C_drone, P_world, R_cam, is_torch = False)
    x_ = K@P_camera
    x = np.copy(x_)

    z = x[2,:]

    x[0,:] = x[0,:]/z
    x[1,:] = x[1,:]/z
    x = x[0:2, :]

    heatmaps = 0

    return x, heatmaps

def take_bone_projection_pytorch(P_world, R_drone, C_drone, R_cam):
    P_camera = world_to_camera(R_drone, C_drone, P_world, R_cam)
    x_ = torch.mm(K_torch, P_camera)
    x = x_.clone()
    
    z = x[2,:]
    
    num_of_joints = P_world.data.shape[1]
    result = Variable(torch.zeros([2, num_of_joints]), requires_grad = False)
    result[0,:] = x[0,:]/z
    result[1,:] = x[1,:]/z
    
    heatmaps = 0

    return result, heatmaps

def take_bone_backprojection(bone_pred, R_drone, C_drone, joint_names):
    TORSO_SIZE_ = DEFAULT_TORSO_SIZE
    img_torso_size = np.linalg.norm(bone_pred[:, joint_names.index('neck')] - bone_pred[:, joint_names.index('spine1')])
    if (img_torso_size == 0):
        img_torso_size = 50
    z_val = (FOCAL_LENGTH * TORSO_SIZE_) / img_torso_size

    bone_pos_3d = np.zeros([3, bone_pred.shape[1]])
    bone_pos_3d[0,:] = bone_pred[0,:]*z_val
    bone_pos_3d[1,:] = bone_pred[1,:]*z_val
    bone_pos_3d[2,:] = z_val
    
    bone_pos_3d = K_inv.dot(bone_pos_3d)
    P_world = camera_to_world (R_drone, C_drone, bone_pos_3d, is_torch= False)
    
    return P_world

def take_bone_backprojection_pytorch(bone_pred, R_drone, C_drone, joint_names):
    num_of_joints = bone_pred.data.shape[1]
    TORSO_SIZE_ = DEFAULT_TORSO_SIZE

    ones_tensor = Variable(torch.ones([1, num_of_joints]), requires_grad=False)*1.0
    img_torso_size = torch.norm(bone_pred[:, joint_names.index('neck')] - bone_pred[:, joint_names.index('spine1')])
    z_val = ((FOCAL_LENGTH * TORSO_SIZE_) / img_torso_size)

    bone_pos_3d = Variable(torch.zeros([3, num_of_joints]))
    bone_pos_3d[0,:] = bone_pred[0,:]*z_val
    bone_pos_3d[1,:] = bone_pred[1,:]*z_val
    bone_pos_3d[2,:] = ones_tensor*z_val
    
    bone_pos_3d = torch.mm(K_inv_torch, bone_pos_3d)
    P_world = camera_to_world (R_drone, C_drone, bone_pos_3d)

    return P_world

def world_to_camera(R_drone, C_drone, P_world, R_cam, is_torch = True):
    if is_torch == True:
        P_drone = torch.mm(torch.inverse(torch.cat((torch.cat((R_drone, C_drone), 1), neat_tensor), 0) ), torch.cat((P_world, ones_tensor), 0) )
        P_camera = torch.mm(torch.inverse(torch.cat((torch.cat((R_cam, C_cam_torch), 1), neat_tensor), 0) ), P_drone)
        P_camera = P_camera[0:3,:]
        P_camera = torch.mm(FLIP_X_Y_torch, P_camera)

    else:
        P_drone = np.linalg.inv(np.vstack([np.hstack([R_drone, C_drone]), np.array([[0,0,0,1]])]))@np.vstack([P_world,  np.ones([1, P_world.shape[1]]) ] )
        P_camera =  np.linalg.inv(np.vstack([np.hstack([R_cam, C_cam]), np.array([[0,0,0,1]])]))@P_drone
        P_camera = P_camera[0:3,:]
        P_camera = FLIP_X_Y@P_camera

    return P_camera    

def camera_to_world(R_drone, C_drone, P_camera, is_torch = True):
    if is_torch == True:
        R_cam_np = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, pi/2, CAMERA_YAW_OFFSET, returnTensor = False)
        R_cam = Variable(torch.from_numpy(R_cam_np).float(), requires_grad = False)

        num_of_joints = P_camera.data.shape[1]

        ones_tensor = Variable(torch.ones([1, num_of_joints]), requires_grad=False)*1.0
        P_camera = torch.mm(FLIP_X_Y_inv_torch, P_camera)
        P_camera = torch.cat((P_camera, ones_tensor),0)
        P_drone = torch.mm(torch.cat((R_cam_torch, C_cam_torch),1), P_camera)
        P_world_ = torch.mm(torch.cat((R_drone, C_drone), 1) ,torch.cat((P_drone, ones_tensor),0))
        P_world = P_world_.clone()

    else:
        R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, pi/2, CAMERA_YAW_OFFSET, returnTensor = False)

        num_of_joints = P_camera.shape[1]

        P_camera = FLIP_X_Y_inv.dot(P_camera)
        P_camera = np.vstack([P_camera, np.ones([1,P_camera.shape[1]]) ])
        P_drone = np.hstack([R_cam, C_cam]).dot(P_camera)
        P_world_ = np.hstack([R_drone, C_drone]).dot(np.vstack([P_drone, np.ones([1, num_of_joints])]))
        P_world = np.copy(P_world_)

    return P_world    


def transform_cov_matrix(R_drone, cov_):
    transformed_cov = (R_drone@R_cam)@cov_@(R_drone@R_cam).T
    return transformed_cov

def take_potential_projection(potential_state, future_pose):
    C_drone = potential_state["position"]
    C_drone = C_drone[:, np.newaxis]
    yaw = potential_state["orientation"]
    R_drone = euler_to_rotation_matrix(0,0,yaw)
    camera_pitch = potential_state["pitch"]
    R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, camera_pitch+pi/2, CAMERA_YAW_OFFSET, returnTensor = False)
    proj, _ =  take_bone_projection(future_pose, R_drone, C_drone, R_cam)
    return proj

class Projection_Client(object):
    def reset(self, data_list, num_of_joints):
        self.num_of_joints = num_of_joints
        self.window_size = len(data_list)
        self.drone_transformation = torch.zeros(self.window_size , 4, 4)

        queue_index = 0
        self.pose_2d_tensor = torch.zeros(self.window_size , 2, num_of_joints)
        for bone_2d_, R_drone_, C_drone_ in data_list:
            R_drone_torch = torch.from_numpy(R_drone_).float()
            C_drone_torch = torch.from_numpy(C_drone_).float()
            self.pose_2d_tensor[queue_index, :, :] = torch.from_numpy(bone_2d_).float()
            self.drone_transformation[queue_index, :, :]= torch.inverse(torch.cat((torch.cat((R_drone_torch, C_drone_torch), dim=1), neat_tensor), dim=0) )
            queue_index += 1

        R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, pi/2, CAMERA_YAW_OFFSET, returnTensor = True)
        temp = torch.inverse(torch.cat((torch.cat((R_cam, C_cam_torch), dim=1), neat_tensor), dim=0) )
        self.camera_transformation = temp.repeat(self.window_size , 1, 1)
        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints)
        self.flip_x_y_tensor = (torch.cat((FLIP_X_Y_torch, torch.zeros(3,1)), dim=1)).repeat(self.window_size , 1, 1)
        self.camera_intrinsics = K_torch.repeat(self.window_size , 1,1)


    def reset_future(self, data_list, num_of_joints, R_cam, R_drone, C_drone, potential_projected_est):
        self.num_of_joints = num_of_joints
        self.window_size = len(data_list)+1
        self.drone_transformation = torch.zeros(self.window_size , 4, 4)
        self.pose_2d_tensor = torch.zeros(self.window_size , 2, num_of_joints)

        self.pose_2d_tensor[0, :, :] = potential_projected_est
        self.drone_transformation[0, :, :]= torch.inverse(torch.cat((torch.cat((R_drone, C_drone), dim=1), neat_tensor), dim=0))

        queue_index = 1
        for bone_2d_, R_drone_, C_drone_ in data_list:
            R_drone_torch = torch.from_numpy(R_drone_).float()
            C_drone_torch = torch.from_numpy(C_drone_).float()
            self.pose_2d_tensor[queue_index, :, :] = torch.from_numpy(bone_2d_).float()
            self.drone_transformation[queue_index, :, :]= torch.inverse(torch.cat((torch.cat((R_drone_torch, C_drone_torch), dim=1), neat_tensor), dim=0) )
            queue_index += 1

        R_cam_normal = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, pi/2, CAMERA_YAW_OFFSET, returnTensor = True)
        self.camera_transformation = (torch.inverse(torch.cat((torch.cat((R_cam_normal, C_cam_torch), dim=1), neat_tensor), dim=0))).repeat(self.window_size-1, 1, 1)
        temp = torch.inverse(torch.cat((torch.cat((R_cam, C_cam_torch), dim=1), neat_tensor), dim=0)).unsqueeze(0)
        self.camera_transformation = torch.cat((temp, self.camera_transformation), dim=0)

        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints)
        self.flip_x_y_tensor = (torch.cat((FLIP_X_Y_torch, torch.zeros(3,1)), dim=1)).repeat(self.window_size , 1, 1)
        self.camera_intrinsics = K_torch.repeat(self.window_size , 1,1)

        self.ones_tensor = torch.ones(self.window_size, 1, self.num_of_joints)

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

