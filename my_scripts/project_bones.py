from helpers import *
import torch
from torch.autograd import Variable
from math import pi, cos, sin, degrees
import numpy as np

neat_tensor = Variable(torch.FloatTensor([[0, 0, 0, 1]]), requires_grad=False) #this tensor is neat!
DEFAULT_TORSO_SIZE = 0.436*0.60#0.86710678118

def euler_to_rotation_matrix(roll, pitch, yaw, returnTensor=False):
    if (returnTensor == True):
        return torch.FloatTensor([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])
    return np.array([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])

R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, CAMERA_PITCH_OFFSET+pi/2, CAMERA_YAW_OFFSET, returnTensor = False)
C_cam = np.array([[CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z]]).T
R_cam_torch = Variable(torch.from_numpy(R_cam).float(), requires_grad = False)
C_cam_torch = Variable(torch.FloatTensor([[CAMERA_OFFSET_X], [CAMERA_OFFSET_Y], [CAMERA_OFFSET_Z]]), requires_grad = False)

FLIP_X_Y = np.array([[0,1,0],[-1,0,0],[0,0,1]])
FLIP_X_Y_torch = Variable(torch.FloatTensor([[0,1,0],[-1,0,0],[0,0,1]]), requires_grad = False)

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

def take_bone_projection(P_world, R_drone, C_drone):
    P_camera = world_to_camera(R_drone, C_drone, P_world, is_torch = False)

    x_ = K@P_camera
    x = np.copy(x_)

    z = x[2,:]

    x[0,:] = x[0,:]/z
    x[1,:] = x[1,:]/z
    x = x[0:2, :]

    heatmaps = 0

    return x, heatmaps

def take_bone_projection_pytorch(P_world, R_drone, C_drone):
    P_camera = world_to_camera(R_drone, C_drone, P_world)
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

def world_to_camera(R_drone, C_drone, P_world, is_torch = True):
    if is_torch == True:
        num_of_joints = P_world.data.shape[1]
        ones_tensor = Variable(torch.ones([1, num_of_joints]), requires_grad=False)*1.0

        P_drone = torch.mm(torch.inverse(torch.cat((torch.cat((R_drone, C_drone), 1), neat_tensor), 0) ), torch.cat((P_world, ones_tensor), 0) )
        P_camera = torch.mm(torch.inverse(torch.cat((torch.cat((R_cam_torch, C_cam_torch), 1), neat_tensor), 0) ), P_drone)
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
        num_of_joints = P_camera.data.shape[1]

        ones_tensor = Variable(torch.ones([1, num_of_joints]), requires_grad=False)*1.0
        P_camera = torch.mm(FLIP_X_Y_inv_torch, P_camera)
        P_camera = torch.cat((P_camera, ones_tensor),0)
        P_drone = torch.mm(torch.cat((R_cam_torch, C_cam_torch),1), P_camera)
        P_world_ = torch.mm(torch.cat((R_drone, C_drone), 1) ,torch.cat((P_drone, ones_tensor),0))
        P_world = P_world_.clone()

    else:
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

