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
import os


THETA_LIST = list(range(270, 180, -20))
PHI_LIST = list(range(0, 360, 20))
hip_index = joint_names_mpi.index('spine1')
num_of_clusters = 8
num_of_anim = 18


USE = 88

dir_name = "num_clusters_" + str(num_of_clusters)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def dome_experiment(pose):
    new_radius = SAFE_RADIUS
        
    #find human orientation (from GT)
    shoulder_vector = pose[:, joint_names_mpi.index('left_arm')] - pose[:, joint_names_mpi.index('right_arm')] 
    human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])
    drone_pos_arr = np.zeros([3,USE])
    ind = 0
    for new_theta_deg in THETA_LIST:
        for new_phi_deg in PHI_LIST:
            new_theta = radians(new_theta_deg)
            new_phi = radians(new_phi_deg)

            #find coordinates of drone
            if ind != 0:
                new_yaw = new_phi  + human_orientation
                drone_pos_arr[0, ind-(90-USE)] = new_radius*cos(new_yaw)*sin(new_theta) + pose[0, hip_index]
                drone_pos_arr[1, ind-(90-USE)] = new_radius*sin(new_yaw)*sin(new_theta) + pose[1, hip_index]
                drone_pos_arr[2, ind-(90-USE)] = new_radius*cos(new_theta)+ pose[2, hip_index]
            ind += 1
    return drone_pos_arr

def vis(pose, drone_pos_arr, errors, custom_name = "", minimum=-1, maximum=-1):
    
    fig = plt.figure( figsize=(10, 4))
    ax = fig.add_subplot(131, projection='3d')

    X = drone_pos_arr[0, :]
    Y = drone_pos_arr[1, :]
    Z = -drone_pos_arr[2, :]

    cmap = cm.cool
    if minimum != -1:
        norm = colors.Normalize(vmin=np.floor(minimum), vmax=np.ceil(maximum))
    else:
        norm = colors.Normalize(vmin=np.floor(np.min(errors)), vmax=np.ceil(np.max(errors)))
    plotd = ax.scatter(X, Y, Z, c=errors, cmap=cmap, norm=norm, marker="^", s=50, alpha=1)

    for i, bone in enumerate(bones_mpi):
        plot1, = ax.plot(pose[0,bone], pose[1,bone], -pose[2,bone], c='xkcd:light blue')
    
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=30., azim=135)

    ax = fig.add_subplot(132)

    plotd = ax.scatter(X, Y, c=errors, cmap=cmap, norm=norm, marker="^", s=50, alpha=1)
    ind =0
    for phi in PHI_LIST:
        if ind >= 90-USE:
            plt.text(X[ind-(90-USE)], Y[ind-(90-USE)], s=str(phi), horizontalalignment="center", color="black" )
        ind +=1

    #max_range = np.array([X.max()-X.min(), Y.max()-Y.min()]).max() *0.3
    plt.xticks([])
    plt.yticks([])

    #ax = fig.add_subplot(143)

    #im_errors = np.hstack([errors, np.zeros([90-USE,])])
   # im_errors = im_errors.reshape([len(THETA_LIST), len(PHI_LIST)])

   # im = ax.imshow(im_errors, cmap=cmap, norm=norm)

   # ind = 0
   # for i in range(len(THETA_LIST)):
     #   for j in range(len(PHI_LIST)):
            #if ind >= 90-USE:
              #  plt.text(j, i, format(im_errors[i, j], '.2f'), horizontalalignment="center", color="black" )
            #ind +=1
    #plt.yticks(np.arange(len(THETA_LIST)), THETA_LIST)
    #plt.ylabel('Theta (Latitude)')
    #plt.xticks(np.arange(len(PHI_LIST)), PHI_LIST)
    #plt.xlabel('Phi (Longitude)')

    ax = fig.add_subplot(133,  projection='3d')
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

    #plt.show()
    fig.subplots_adjust(right=0.8, hspace = 0.5)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(plotd, cax=cbar_ax, shrink = 0.8)
    plt.savefig(dir_name + '/openpose_err_' + custom_name +'.png')
    #plt.show()
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

    #plt.show()
    plt.close(fig)

def display_pose_clusters(pose_clusters, cluster_num=num_of_clusters, custom_name = ""):
    if cluster_num < 9:
        fig = plt.figure(figsize=(4*num_of_clusters, 4))
    else: 
        fig = plt.figure(figsize=(int(4*cluster_num/2), 8))
    for cluster in range(cluster_num):
        if cluster_num < 9:
            ax = fig.add_subplot(1,cluster_num,cluster+1,  projection='3d')
        else:
            ax = fig.add_subplot(2,np.ceil(cluster_num/2),cluster+1,  projection='3d')

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

    #plt.show()
    plt.savefig(dir_name + '/clusters' + custom_name +'.png')
    plt.close(fig)


#read openpose file
    #store values
    #store poses           
openpose_err = pd.read_csv("openpose_error.txt", sep='\t', skiprows=1, header=None).iloc[:,:-1].values.astype('float')
error_values = openpose_err[:,(90-USE):90]
pose_values = openpose_err[:,90:]

arm_error_values = pd.read_csv("openpose_arm_error.txt", sep='\t', header=None).iloc[1:,(90-USE):-1].values.astype('float')                
leg_error_values = pd.read_csv("openpose_leg_error.txt", sep='\t', header=None).iloc[1:,(90-USE):-1].values.astype('float')     

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
shoulder_vector = normalized_poses[:, 0:2, joint_names_mpi.index('spine1')] - normalized_poses[:, 0:2, joint_names_mpi.index('right_arm')] 
shoulder_vector = shoulder_vector/np.linalg.norm(shoulder_vector, axis=1)[:, np.newaxis]
rotated_poses =  np.zeros(poses.shape)
rotated_poses[:,2,:] =  normalized_poses[:,2,:].copy()

align_vec = np.array([1, 1])/np.linalg.norm(np.array([1, 1]))

for pose_ind in range(shoulder_vector.shape[0]):
    angle_between_them = acos(np.dot(shoulder_vector[pose_ind, :], align_vec))
    rotated_poses[pose_ind,0,:] = cos(angle_between_them) * normalized_poses[pose_ind,0,:] - sin(angle_between_them) * normalized_poses[pose_ind,1,:]
    rotated_poses[pose_ind,1,:] = sin(angle_between_them) * normalized_poses[pose_ind,0,:] + cos(angle_between_them) * normalized_poses[pose_ind,1,:] 
    
    new_shoulder_vector = rotated_poses[pose_ind, 0:2, joint_names_mpi.index('spine1')] - rotated_poses[pose_ind, 0:2, joint_names_mpi.index('right_arm')]
    new_shoulder_vector = new_shoulder_vector/np.linalg.norm(new_shoulder_vector)
    angle_between_them_new = acos(np.dot(new_shoulder_vector, align_vec))

    if angle_between_them_new > 1e-5:
        rotated_poses[pose_ind,0,:] = cos(-angle_between_them) * normalized_poses[pose_ind,0,:] - sin(-angle_between_them) * normalized_poses[pose_ind,1,:]
        rotated_poses[pose_ind,1,:] = sin(-angle_between_them) * normalized_poses[pose_ind,0,:] + cos(-angle_between_them) * normalized_poses[pose_ind,1,:] 

    new_shoulder_vector = rotated_poses[pose_ind, 0:2, joint_names_mpi.index('spine1')] - rotated_poses[pose_ind, 0:2, joint_names_mpi.index('right_arm')]
    new_shoulder_vector = new_shoulder_vector/np.linalg.norm(new_shoulder_vector)
    angle_between_them_new = acos(np.dot(new_shoulder_vector, align_vec))
    #display_pose(rotated_poses[pose_ind, :, :])

reshaped_poses = rotated_poses.reshape([-1, 45], order="F")

kmeans = KMeans(n_clusters=num_of_clusters, random_state=0)
kmeans.fit(reshaped_poses)
cluster_centers = (kmeans.cluster_centers_).reshape([-1, 3, 15], order="F")
display_pose_clusters(cluster_centers)
labels = kmeans.labels_

mean_err_cluster = np.zeros([num_of_clusters,USE])
mean_err_arm_cluster = np.zeros([num_of_clusters,USE])
mean_err_leg_cluster = np.zeros([num_of_clusters,USE])

for clusters in range(num_of_clusters):
    mean_err_cluster[clusters, :] = np.mean(error_values[labels == clusters], axis=0)
    mean_err_arm_cluster[clusters, :] = np.mean(arm_error_values[labels == clusters], axis=0)
    mean_err_leg_cluster[clusters, :] = np.mean(leg_error_values[labels == clusters], axis=0)

for clusters in range(num_of_clusters):
    drone_pos_arr = dome_experiment(cluster_centers[clusters, :, :])
    #vis(cluster_centers[clusters, :, :], drone_pos_arr, mean_err_cluster[clusters, :], custom_name="cluster_" + str(clusters), maximum=np.max(mean_err_cluster), minimum=np.min(mean_err_cluster))
    #vis(cluster_centers[clusters, :, :], drone_pos_arr, mean_err_arm_cluster[clusters, :], custom_name="cluster_arm_" + str(clusters), maximum=np.max(mean_err_arm_cluster), minimum=np.min(mean_err_arm_cluster))
    #vis(cluster_centers[clusters, :, :], drone_pos_arr, mean_err_leg_cluster[clusters, :], custom_name="cluster_leg_" + str(clusters), maximum=np.max(mean_err_leg_cluster), minimum=np.min(mean_err_leg_cluster))

######### cluster by animation
mean_err_cluster_anim  = np.zeros([num_of_anim,USE])
mean_poses_anim = np.zeros([num_of_anim,3, 15])
for animation in range(num_of_anim):
    mean_poses_anim[animation, :, :] = np.mean(rotated_poses[animation*60:(animation+1)*60, :, :], axis=0)
    mean_err_cluster_anim[animation, :] = np.mean(error_values[animation*60:(animation+1)*60], axis=0)
display_pose_clusters(mean_poses_anim, num_of_anim, custom_name="_anim")

for animation in range(num_of_anim):
    drone_pos_arr = dome_experiment(mean_poses_anim[animation, :, :])
    vis(mean_poses_anim[animation, :, :], drone_pos_arr, mean_err_cluster_anim[animation, :], custom_name="anim_cluster_" + str(animation), maximum=np.max(mean_err_cluster_anim), minimum=np.min(mean_err_cluster_anim))
