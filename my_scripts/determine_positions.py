from helpers import * 
from State import *
from NonAirSimClient import *
from pose3d_optimizer import *
from pose3d_optimizer_scipy import *
from project_bones import *
import numpy as np
import cv2 as cv
import torch as torch
import time
from scipy.optimize import least_squares
import pdb
import util as demo_util
from PoseEstimationClient import *

#import openpose as openpose_module
#import liftnet as liftnet_module

objective_online = pose3d_online_parallel_wrapper()
objective_calib = pose3d_calibration_parallel_wrapper()

def adjust_with_M(M, pose, hip_index):
    root_pose = pose[:, hip_index]
    return np.dot(pose - root_pose[:, np.newaxis], M)+root_pose[:, np.newaxis]

def determine_all_positions(airsim_client, pose_client, current_state, plot_loc = 0, photo_loc = 0):
    if (pose_client.modes["mode_3d"] == 0):
        determine_3d_positions_all_GT(airsim_client, pose_client, current_state, plot_loc, photo_loc)
    elif (pose_client.modes["mode_3d"] == 1): 
        determine_3d_positions_backprojection(airsim_client, pose_client, current_state, plot_loc, photo_loc)
    elif (pose_client.modes["mode_3d"] == 3):
        determine_3d_positions_energy_scipy(airsim_client, pose_client, current_state, plot_loc, photo_loc)
    current_state.update_human_info(pose_client.current_pose)

def determine_2d_positions(pose_client, current_state, return_heatmaps=True, is_torch = True, input_image = 0,  scales = [1]):
    mode_2d, cropping_tool = pose_client.modes["mode_2d"], pose_client.cropping_tool

    bone_2d_gt, heatmaps = find_2d_pose_gt (current_state=current_state, input_image=input_image, cropping_tool=cropping_tool, return_heatmaps=return_heatmaps, is_torch=is_torch)
    if (mode_2d != 2):
        bone_2d = bone_2d_gt.clone()
        if (mode_2d == 1):
            bone_2d = pose_client.add_2d_noise(bone_2d)
        heatmaps_scales = 0
        poses_scales = 0
    else:            
        bone_2d, heatmaps, heatmaps_scales, poses_scales = find_2d_pose_openpose(input_image,  scales)

    pose_client.openpose_error = np.mean(np.linalg.norm(bone_2d_gt-bone_2d, axis=0))
    if not pose_client.USE_SINGLE_JOINT:
        arm_joints, _, _ = return_arm_joints()
        leg_joints, _, _ = return_leg_joints()
        pose_client.openpose_arm_error = np.mean(np.linalg.norm(bone_2d_gt[:, arm_joints]-bone_2d[:, arm_joints], axis=0))
        pose_client.openpose_leg_error = np.mean(np.linalg.norm(bone_2d_gt[:, leg_joints]-bone_2d[:, leg_joints], axis=0))
    return bone_2d, heatmaps, heatmaps_scales, poses_scales

def find_2d_pose_gt(current_state, input_image, cropping_tool, return_heatmaps=True, is_torch=True):
    bone_pos_3d_GT, R_drone_gt, C_drone_gt, R_cam_gt = current_state.get_frame_parameters()

    bone_2d_var, heatmaps = take_bone_projection_pytorch(torch.from_numpy(bone_pos_3d_GT).float(), R_drone_gt, C_drone_gt, R_cam_gt)
    
    bone_2d = bone_2d_var.detach()
    if (return_heatmaps):
        bone_2d = cropping_tool.crop_pose(bone_2d)
        heatmaps = create_heatmap(bone_2d.data.cpu().numpy(), input_image.shape[1], input_image.shape[0])
    else:
        heatmaps = 0
    return bone_2d, heatmaps

def find_2d_pose_openpose(input_image, scales):
    poses, heatmaps, heatmaps_scales, poses_scales = openpose_module.run_only_model(input_image, scales)
    return poses, heatmaps.cpu().numpy(), heatmaps_scales, poses_scales


def determine_relative_3d_pose(mode_lift, current_state, bone_2d, cropped_image, heatmap_2d):
    bone_pos_3d_GT, R_drone_gt, C_drone_gt, R_cam_gt = current_state.get_frame_parameters()

    if (mode_lift == 0):
        pose3d_relative = torch.from_numpy(bone_pos_3d_GT).clone()

    elif (mode_lift  == 1):
        num_of_joints = bone_2d.shape[1]
        pose = torch.cat((torch.t(bone_2d), torch.ones(num_of_joints,1)), 1)
        pose3d_lift, _, _ = liftnet_module.run(cropped_image, heatmap_2d, pose)
        pose3d_lift = pose3d_lift.view(num_of_joints+2,  -1).permute(1, 0)
        pose3d_lift = rearrange_bones_to_mpi(pose3d_lift, is_torch=True)
        pose3d_relative = camera_to_world(R_drone_gt, C_drone_gt, R_cam_gt, pose3d_lift.cpu())
    return pose3d_relative

def initialize_with_gt(airsim_client, pose_client, current_state, plot_loc = 0, photo_loc = 0):
    bone_pos_3d_GT, R_drone_gt, C_drone_gt, R_cam_gt = current_state.get_frame_parameters()
    bone_connections, joint_names, num_of_joints, _ = pose_client.model_settings()
    
    input_image = cv.imread(photo_loc)
    cropped_image, scales = pose_client.cropping_tool.crop(input_image, airsim_client.linecount)

    #find 2d pose (using openpose or gt)
    bone_2d, heatmap_2d, _, _ = determine_2d_positions(pose_client=pose_client, current_state=current_state, return_heatmaps=True, is_torch=True, input_image=cropped_image, scales=scales)

    #find relative 3d pose using liftnet or GT relative pose
    if not pose_client.USE_SINGLE_JOINT:
        pose3d_lift = determine_relative_3d_pose(pose_client.modes["mode_lift"], current_state, bone_2d, cropped_image, heatmap_2d)
        pose3d_lift_directions = calculate_bone_directions(pose3d_lift, np.array(return_lift_bone_connections(bone_connections)), batch=False) 
    else:
        pose3d_lift_directions = None

    #uncrop 2d pose
    bone_2d = pose_client.cropping_tool.uncrop_pose(bone_2d)

    #add information you need to your window
    if pose_client.isCalibratingEnergy:
        pose_client.addNewFrame(bone_2d, R_drone_gt, C_drone_gt, R_cam_gt, airsim_client.linecount, pose3d_lift_directions)
        P_world = bone_pos_3d_GT
    else:
        for _ in range(pose_client.ONLINE_WINDOW_SIZE):
            pose_client.addNewFrame(bone_2d, R_drone_gt, C_drone_gt, R_cam_gt, airsim_client.linecount, pose3d_lift_directions)
        P_world = np.repeat(bone_pos_3d_GT[np.newaxis, :, :], pose_client.ONLINE_WINDOW_SIZE+1, axis=0)
    pose_client.update3dPos(P_world, P_world.copy())
    if not pose_client.USE_SINGLE_JOINT:
        pose_client.update_bone_lengths(torch.from_numpy(bone_pos_3d_GT).float())
    pose_client.future_pose = current_state.bone_pos_gt
    pose_client.current_pose = current_state.bone_pos_gt
    
    if pose_client.USE_TRAJECTORY_BASIS:
        pose_client.optimized_traj[0,:,:] = current_state.bone_pos_gt.copy()


def determine_openpose_error(airsim_client, pose_client, current_state, plot_loc = 0, photo_loc = 0):
    bone_pos_3d_GT, _, C_drone_gt, _ = current_state.get_frame_parameters()
    bone_connections, _, num_of_joints, _ =  pose_client.model_settings()

    input_image = cv.imread(photo_loc)
    cropped_image, scales = pose_client.cropping_tool.crop(input_image, airsim_client.linecount)
    #save_image(cropped_image, airsim_client.linecount, plot_loc)
    bone_2d, _, _, _ = determine_2d_positions(pose_client=pose_client, current_state=current_state, return_heatmaps=True, is_torch=True, input_image=cropped_image, scales=scales)
    bone_2d = pose_client.cropping_tool.uncrop_pose(bone_2d)
    #superimpose_on_image(bone_2d.numpy(), plot_loc, airsim_client.linecount, bone_connections, photo_loc, custom_name="projected_res_", scale = -1)

    pose_client.future_pose = bone_pos_3d_GT
    pose_client.current_pose = bone_pos_3d_GT

    plot_end = {"est": bone_pos_3d_GT, "GT": bone_pos_3d_GT, "drone": C_drone_gt, "eval_time": 0, "f_string": ""}
    pose_client.append_res(plot_end)
    pose_client.f_reconst_string = "" 

def determine_3d_positions_energy_scipy(airsim_client, pose_client, current_state, plot_loc=0, photo_loc=0):
    bone_pos_3d_GT, R_drone_gt, C_drone_gt, R_cam_gt = current_state.get_frame_parameters()
    bone_connections, joint_names, num_of_joints, hip_index = pose_client.model_settings()

    input_image = cv.imread(photo_loc)
    cropped_image, scales = pose_client.cropping_tool.crop(input_image, airsim_client.linecount)

    #find 2d pose (using openpose or gt)
    bone_2d, heatmap_2d, _, _ = determine_2d_positions(pose_client=pose_client, current_state=current_state, return_heatmaps=True, is_torch=True, input_image=cropped_image, scales=scales)

    #find relative 3d pose using liftnet or GT relative pose
    if not pose_client.USE_SINGLE_JOINT:
        pose3d_lift = determine_relative_3d_pose(pose_client.modes["mode_lift"], current_state, bone_2d, cropped_image, heatmap_2d)
        pose3d_lift_directions = calculate_bone_directions(pose3d_lift, np.array(return_lift_bone_connections(bone_connections)), batch=False) 
    else:
        pose3d_lift_directions = None

    #uncrop 2d pose
    bone_2d = pose_client.cropping_tool.uncrop_pose(bone_2d)

    #add current pose as initial pose. if first frame, take backprojection for initialization
    if (airsim_client.linecount != 0):
        pre_pose_3d = pose_client.P_world[FUTURE_POSE_INDEX, :, :].copy()
    else:
        if pose_client.init_pose_with_gt:
            pre_pose_3d = bone_pos_3d_GT.copy()
        else:
            pre_pose_3d = take_bone_backprojection_pytorch(bone_2d, R_drone_gt, C_drone_gt, R_cam_gt, joint_names).numpy()

    #add information you need to your window
    pose_client.addNewFrame(bone_2d, R_drone_gt, C_drone_gt, R_cam_gt, airsim_client.linecount, pose3d_lift_directions)

    final_loss = np.zeros([1,1])
    result_shape = pose_client.result_shape

    if (airsim_client.linecount > 0):
        #calibration mode parameters
        if (pose_client.isCalibratingEnergy): 
            loss_dict = pose_client.loss_dict_calib
            pose3d_init_scrambled = pre_pose_3d
            #noise = pose_client.numpy_random.normal(0, 0.5, pose3d_init_scrambled.shape)
            noisy_init_pose = pose3d_init_scrambled# + noise
            result_size = result_shape[0]*result_shape[1]
            pose3d_init = np.reshape(a = noisy_init_pose, newshape = [result_size,], order = "C")
            objective = objective_calib
            objective_jacobian =  objective_calib.jacobian

        #online mode parameters
        else:
            loss_dict = pose_client.loss_dict_online
            result_size = result_shape[0]*result_shape[1]*result_shape[2]
            if pose_client.USE_TRAJECTORY_BASIS:
                pose3d_init = pose_client.optimized_traj.copy()
                pose3d_init = np.reshape(a = pose3d_init, newshape = [result_size], order = "C")
            else:
                pose3d_init = pose_client.P_world.copy()
                pose3d_init = np.reshape(a = pose3d_init, newshape = [result_size,], order = "C")
            objective = objective_online
            objective_jacobian = objective_online.jacobian

        objective.reset(pose_client)
        start_time = time.time()
        optimized_res = least_squares(objective.forward, pose3d_init, jac=objective_jacobian, bounds=(-np.inf, np.inf), method=pose_client.method, ftol=pose_client.ftol)
        func_eval_time = time.time() - start_time
        print("least squares eval time", func_eval_time)
        if pose_client.USE_TRAJECTORY_BASIS:
            optimized_traj = np.reshape(a = optimized_res.x, newshape = result_shape, order = "C")
            P_world = project_trajectory(torch.from_numpy(optimized_traj).float(), pose_client.ONLINE_WINDOW_SIZE, pose_client.NUMBER_OF_TRAJ_PARAM).numpy()
            pose_client.optimized_traj = optimized_traj
        else:
            P_world = np.reshape(a = optimized_res.x, newshape = result_shape, order = "C")

        if (pose_client.isCalibratingEnergy):
            pose_client.update_bone_lengths(torch.from_numpy(P_world).float())

        pose_client.update3dPos(P_world, bone_pos_3d_GT)

    #if the frame is the first frame, the pose is found through backprojection
    else:
        pose_client.update3dPos(pre_pose_3d, bone_pos_3d_GT)
        loss_dict = pose_client.loss_dict_calib
        func_eval_time = 0
        noisy_init_pose = pre_pose_3d

    pose_client.error_2d.append(final_loss[0])

    adjusted_current_pose = adjust_with_M(pose_client.M, pose_client.current_pose, hip_index)
    #adjusted_future_pose = adjust_with_M(pose_client.M, pose_client.future_pose, hip_index)
    adjusted_middle_pose = adjust_with_M(pose_client.M, pose_client.middle_pose, hip_index)
    middle_pose_GT = pose_client.update_middle_pose_GT(bone_pos_3d_GT)
    check, _ = take_bone_projection_pytorch(torch.from_numpy(pose_client.current_pose).float(), R_drone_gt, C_drone_gt, R_cam_gt)

    #lots of plot stuff
    error_3d = np.mean(np.linalg.norm(bone_pos_3d_GT - adjusted_current_pose, axis=0))
    middle_pose_error = np.mean(np.linalg.norm(middle_pose_GT - adjusted_middle_pose, axis=0))

    pose_client.error_3d.append(error_3d)
    pose_client.middle_pose_error.append(middle_pose_error)
    ave_error =  sum(pose_client.error_3d)/len(pose_client.error_3d)
    ave_middle_error =  sum(pose_client.middle_pose_error)/len(pose_client.middle_pose_error)

    if (plot_loc != 0 and not pose_client.quiet): 
        #superimpose_on_image(bone_2d.numpy(), plot_loc, airsim_client.linecount, bone_connections, photo_loc, custom_name="projected_res_", scale = -1, projection=check.numpy())
        #superimpose_on_image(bone_2d.numpy(), plot_loc, airsim_client.linecount, bone_connections, photo_loc, custom_name="projected_res_2_", scale = -1)

        #plot_2d_projection(check, plot_loc, airsim_client.linecount, bone_connections, custom_name="proj_2d")

        plot_human(bone_pos_3d_GT, adjusted_current_pose, plot_loc, airsim_client.linecount, bone_connections, pose_client.USE_SINGLE_JOINT, error_3d, additional_text = ave_error)
        #plot_human(bone_pos_3d_GT, noisy_init_pose, plot_loc, airsim_client.linecount, bone_connections, 0, custom_name="init_pose", label_names = ["GT", "Init"])

        #save_heatmaps(heatmap_2d, airsim_client.linecount, plot_loc)
        #save_heatmaps(heatmaps_scales.cpu().numpy(), client.linecount, plot_loc, custom_name = "heatmaps_scales_", scales=scales, poses=poses_scales.cpu().numpy(), bone_connections=bone_connections)

        if (not pose_client.isCalibratingEnergy):
            plot_human(bone_pos_3d_GT, adjusted_current_pose, plot_loc, airsim_client.linecount-MIDDLE_POSE_INDEX+1, bone_connections, pose_client.USE_SINGLE_JOINT, middle_pose_error, custom_name="middle_pose_", label_names = ["GT", "Estimate"], additional_text = ave_middle_error)
            #plot_human(adjusted_current_pose, adjusted_future_pose, plot_loc, airsim_client.linecount, bone_connections, error_3d, custom_name="future_plot_", label_names = ["current", "future"])
            #pose3d_lift_normalized, _ = normalize_pose(pose3d_lift, hip_index, is_torch=False)
            #bone_pos_3d_GT_normalized, _ = normalize_pose(bone_pos_3d_GT, hip_index, is_torch=False)
            #adjusted_current_pose_normalized, _ = normalize_pose(adjusted_current_pose, hip_index, is_torch=False)
            #plot_human(bone_pos_3d_GT_normalized, pose3d_lift_normalized, plot_loc, airsim_client.linecount, bone_connections, error_3d, custom_name="lift_res_", label_names = ["GT", "LiftNet"])
            #plot_human(pose3d_lift_normalized, adjusted_current_pose_normalized, plot_loc, airsim_client.linecount, bone_connections, error_3d, custom_name="lift_res_2_", label_names = ["LiftNet", "Estimate"])
            #plot_optimization_losses(objective.pltpts, plot_loc, airsim_client.linecount, loss_dict)

    plot_end = {"est": adjusted_current_pose, "GT": bone_pos_3d_GT, "drone": C_drone_gt, "eval_time": func_eval_time, "f_string": ""}
    pose_client.append_res(plot_end)
    reconstruction_str = ""
    for i in range(num_of_joints):
        reconstruction_str += str(adjusted_current_pose[0,i]) + "\t" + str(adjusted_current_pose[1,i]) + "\t" + str(adjusted_current_pose[2,i]) + "\t"
    pose_client.f_reconst_string = reconstruction_str

def determine_3d_positions_backprojection(airsim_client, pose_client, current_state, plot_loc = 0, photo_loc = 0):
    bone_pos_3d_GT, R_drone_gt, C_drone_gt, R_cam_gt = current_state.get_frame_parameters()
    bone_connections, joint_names, _, hip_index = pose_client.model_settings()

    bone_2d, _, _, _ = determine_2d_positions(pose_client=pose_client, current_state=current_state, return_heatmaps=False, is_torch=True, input_image=cropped_image, scales=scales)

    R_drone = euler_to_rotation_matrix(drone_orientation_gt[0], drone_orientation_gt[1], drone_orientation_gt[2])
    C_drone = drone_pos_gt
    
    P_world = take_bone_backprojection(bone_2d, R_drone, C_drone, hip_index)
    error_3d = np.linalg.norm(bone_pos_3d_GT - P_world)
    pose_client.error_3d.append(error_3d)

    if (plot_loc != 0):
        check, _, _ = take_bone_projection(P_world, R_drone, C_drone)
        superimpose_on_image([check], plot_loc, airsim_client.linecount, bone_connections, photo_loc)
        plot_human(bone_pos_3d_GT, P_world, plot_loc, airsim_client.linecount, bone_connections, error_3d)

    plot_end = {"est": P_world, "GT": bone_pos_3d_GT, "drone": C_drone, "eval_time": 0, "f_string": ""}
    pose_client.append_res(plot_end)


def determine_3d_positions_all_GT(airsim_client, pose_client, current_state, plot_loc, photo_loc):
    bone_pos_3d_GT, R_drone_gt, C_drone_gt, R_cam_gt = current_state.get_frame_parameters()
    bone_connections, joint_names, num_of_joints, _ =  pose_client.model_settings()

    if (pose_client.modes["mode_2d"] == 2):
        input_image = cv.imread(photo_loc) 

        cropped_image, scales = pose_client.cropping_tool.crop(input_image, airsim_client.linecount)
               
        #find 2d pose (using openpose or gt)
        bone_2d, _, _, _ = determine_2d_positions(pose_client=pose_client, current_state=current_state, return_heatmaps=False, is_torch=True, input_image=cropped_image, scales=scales)

        #uncrop 2d pose 
        bone_2d = pose_client.cropping_tool.uncrop_pose(bone_2d)

        if (not pose_client.quiet):
            superimpose_on_image(bone_2d.cpu().numpy(), plot_loc, airsim_client.linecount, bone_connections, photo_loc, custom_name="gt_")
            save_image(cropped_image, airsim_client.linecount, plot_loc, custom_name="cropped_img_")
            save_heatmaps(heatmap_2d, airsim_client.linecount, plot_loc)
            #save_heatmaps(heatmaps_scales.cpu().numpy(), airsim_client.linecount, plot_loc, custom_name = "heatmaps_scales_", scales=scales, poses=poses_scales.cpu().numpy(), bone_connections=bone_connections)

    elif (pose_client.modes["mode_2d"] == 0):
        bone_2d, heatmap_2d, _, _ = determine_2d_positions(pose_client, False, False, unreal_positions, bone_pos_3d_GT, photo_loc, 0)
        if not pose_client.quiet:
            superimpose_on_image(bone_2d, plot_loc, airsim_client.linecount, bone_connections, photo_loc, custom_name="gt_", scale = -1)

    plot_end = {"est": bone_pos_3d_GT, "GT": bone_pos_3d_GT, "drone": C_drone.cpu().numpy(), "eval_time": 0, "f_string": ""}
    pose_client.append_res(plot_end)


def switch_energy(value):
    pass
