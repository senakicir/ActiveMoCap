# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
import sys
sys.path.append("..")
from State import SAFE_RADIUS
from math import radians, cos, sin, acos
from helpers import joint_names_mpi, bones_mpi
from sklearn.cluster import KMeans


THETA_LIST = list(range(270, 180, -20))
PHI_LIST = list(range(0, 360, 20))
hip_index = joint_names_mpi.index('spine1')

def dome_experiment(pose):
    new_radius = SAFE_RADIUS
        
    #find human orientation (from GT)
    shoulder_vector = pose[:, joint_names_mpi.index('left_arm')] - pose[:, joint_names_mpi.index('right_arm')] 
    human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])
    drone_pos_arr = np.zeros([3,87])
    ind = 0
    for new_theta_deg in THETA_LIST:
        for new_phi_deg in PHI_LIST:
            new_theta = radians(new_theta_deg)
            new_phi = radians(new_phi_deg)

            #find coordinates of drone
            if ind != 0:
                new_yaw = new_phi  + human_orientation
                drone_pos_arr[0, ind-3] = new_radius*cos(new_yaw)*sin(new_theta) + pose[0, hip_index]
                drone_pos_arr[1, ind-3] = new_radius*sin(new_yaw)*sin(new_theta) + pose[1, hip_index]
                drone_pos_arr[2, ind-3] = new_radius*cos(new_theta)+ pose[2, hip_index]
            ind += 1
    return drone_pos_arr

def vis(pose, drone_pos_arr, errors):
    
    fig = plt.figure( figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')

    X = drone_pos_arr[0, :]
    Y = drone_pos_arr[1, :]
    Z = -drone_pos_arr[2, :]

    plotd = ax.scatter(X, Y, Z, c=errors, cmap="cool", marker="^", s=50, alpha=1)
    fig.colorbar(plotd, orientation='vertical', shrink=0.8)

    for i, bone in enumerate(bones_mpi):
        plot1, = ax.plot(pose[0,bone], pose[1,bone], -pose[2,bone], c='xkcd:light blue')
    
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax = fig.add_subplot(122,  projection='3d')
    X = pose[0,:]
    Y = pose[1,:]
    Z = -pose[2,:]
    for i, bone in enumerate(bones_mpi):
        plot1, = ax.plot(X[bone], Y[bone], Z[bone], c='xkcd:black')
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    plt.savefig('openpose_err.png')
    plt.close(fig)
   
def display_pose(pose):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111,  projection='3d')
    X = pose[0,:]
    Y = pose[1,:]
    Z = -pose[2,:]
    for _, bone in enumerate(bones_mpi):
        ax.plot(X[bone], Y[bone], Z[bone], c='xkcd:black')

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    plt.close(fig)

def display_pose_clusters(pose_clusters):
    fig = plt.figure(figsize=(32, 4))
    for cluster in range(8):
        ax = fig.add_subplot(1,8,cluster+1,  projection='3d')
        pose = pose_clusters[cluster,:,:]
        X = pose[0,:]
        Y = pose[1,:]
        Z = -pose[2,:]
        for _, bone in enumerate(bones_mpi):
            ax.plot(X[bone], Y[bone], Z[bone], c='xkcd:black')

        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    plt.savefig('clusters.png')
    plt.close(fig)


#read openpose file
    #store values
    #store poses           
openpose_err = pd.read_csv("openpose_error.txt", sep='\t', skiprows=1, header=None).iloc[:,:-1].values.astype('float')
error_values = openpose_err[:,3:90]
pose_values = openpose_err[:,90:]

arm_error_values = pd.read_csv("openpose_arm_error.txt", sep='\t', header=None).iloc[1:,:-1].values.astype('float')                
leg_error_values = pd.read_csv("openpose_leg_error.txt", sep='\t', header=None).iloc[1:,:-1].values.astype('float')     

#find error mean, var
mean_err = np.mean(error_values, axis=0)
std_err = np.mean(error_values, axis=0)

one_pose = pose_values[0, :].reshape([3,-1], order="F")
drone_pos_arr = dome_experiment(one_pose)
vis(one_pose, drone_pos_arr, mean_err)


#######cluster poses###############

#normalize poses
pose_values = openpose_err[:,90:]
poses = pose_values.reshape([-1, 3, 15], order="F")

hip_index = joint_names_mpi.index('spine1')
hip_pose = poses[:, :, hip_index]
normalized_poses = poses - hip_pose[:,:,np.newaxis]

#make everyone's shoulder vector [0,1]
shoulder_vector = normalized_poses[:, :, joint_names_mpi.index('spine1')] - normalized_poses[:, :, joint_names_mpi.index('right_arm')] 
shoulder_vector = shoulder_vector/np.linalg.norm(shoulder_vector, axis=0)
rotated_poses =  np.zeros(poses.shape)
align_vec = np.array([1, 1, 0])/np.linalg.norm(np.array([1, 1, 0]))

for pose_ind in range(shoulder_vector.shape[0]):
    cross = np.cross(align_vec, shoulder_vector[pose_ind, :])
    ab_angle = np.arccos(np.dot(align_vec, shoulder_vector[pose_ind, :]))

    vx = np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
    R = np.identity(3)*np.cos(ab_angle) + (1-np.cos(ab_angle))*np.outer(cross,cross) + np.sin(ab_angle)*vx

    rotated_poses[pose_ind, :] = (np.matmul(normalized_poses[pose_ind, :, :].T, R)).T
    new_shoulder_vector = rotated_poses[pose_ind, :, joint_names_mpi.index('spine1')] - rotated_poses[pose_ind, :, joint_names_mpi.index('right_arm')]
    new_shoulder_vector = new_shoulder_vector / np.linalg.norm(new_shoulder_vector)
    #cross = np.cross(new_shoulder_vector[pose_ind, :], align_vec)
    new_ab_angle = np.arccos(np.dot(new_shoulder_vector, align_vec))
    print(ab_angle, new_ab_angle)

    #angle_between_them = acos(np.dot(shoulder_vector[pose_ind, :], align_vec)
    #rotated_poses[pose_ind,0,:] = cos(angle_between_them) * normalized_poses[pose_ind,0,:] - sin(angle_between_them) * normalized_poses[pose_ind,1,:]
    #rotated_poses[pose_ind,1,:] = sin(angle_between_them) * normalized_poses[pose_ind,0,:] + cos(angle_between_them) * normalized_poses[pose_ind,1,:] 
    
    #new_shoulder_vector = rotated_poses[pose_ind, 0:2, joint_names_mpi.index('spine1')] - rotated_poses[pose_ind, 0:2, joint_names_mpi.index('right_arm')]
    #angle_between_them_new = acos(np.dot(new_shoulder_vector, align_vec) / (np.linalg.norm(new_shoulder_vector)*np.linalg.norm(align_vec)))

    #if angle_between_them_new > 1e-5:
    #    rotated_poses[pose_ind,0,:] = cos(-angle_between_them) * normalized_poses[pose_ind,0,:] - sin(-angle_between_them) * normalized_poses[pose_ind,1,:]
    #    rotated_poses[pose_ind,1,:] = sin(-angle_between_them) * normalized_poses[pose_ind,0,:] + cos(-angle_between_them) * normalized_poses[pose_ind,1,:] 
    
    #new_shoulder_vector = rotated_poses[pose_ind, 0:2, joint_names_mpi.index('spine1')] - rotated_poses[pose_ind, 0:2, joint_names_mpi.index('right_arm')]
    #angle_between_them_new = acos(np.dot(new_shoulder_vector, align_vec) /  (np.linalg.norm(new_shoulder_vector)*np.linalg.norm(align_vec)))
    #display_pose(rotated_poses[pose_ind, :, :])

reshaped_poses = rotated_poses.reshape([-1, 45], order="F")

kmeans = KMeans(n_clusters=8, random_state=0)
kmeans.fit(reshaped_poses)
cluster_centers = (kmeans.cluster_centers_).reshape([-1, 3, 15], order="F")
display_pose_clusters(cluster_centers)
#find error mean, var wrt pose clusters


#visualize error mean, var wrt pose clusters (3d plots)