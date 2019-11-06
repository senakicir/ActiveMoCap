from helpers import * 
from State import *
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
from PoseEstimationClient import PoseEstimationClient
from Lift_Client import calculate_bone_directions, calculate_bone_directions_simple, scale_with_bone_lengths
from yolo_helper import find_single_person_bbox

#import openpose as openpose_module
#import liftnet as liftnet_module
#import darknet as darknet_module

objective_online = pose3d_online_parallel_wrapper()
objective_calib = pose3d_calibration_parallel_wrapper()

def adjust_with_M(M, pose, hip_index):
    if pose.ndim == 2:
        root_pose = pose[:, hip_index]
        return np.dot(pose-root_pose[:,np.newaxis], M)+root_pose[:,np.newaxis]
    elif pose.ndim == 3:
        root_pose = pose[:,:,hip_index]
        return np.matmul(pose-root_pose[:,:,np.newaxis], M) + root_pose[:,:,np.newaxis]

def determine_2d_positions(pose_client, current_state, my_rng, file_manager, linecount=0):    
    photo_loc = file_manager.get_photo_loc()
    pose_2d_gt = find_2d_pose_gt(projection_client=pose_client.projection_client, current_state=current_state)

    if (pose_client.modes["mode_2d"] == "gt" or pose_client.modes["mode_2d"] == "gt_with_noise"):
        bbox = None
    elif (pose_client.modes["mode_2d"] == "openpose"):  
        predictions = darknet_module.detect(photo_loc)
        bbox, yolo_confidence = find_single_person_bbox(predictions)
        file_manager.record_yolo_results(linecount, yolo_confidence, bbox)
    if bbox is None:
        pose_client.cropping_tool.disable_cropping()
    else:
        pose_client.cropping_tool.enable_cropping()

    pose_client.cropping_tool.update_bbox(bbox)
    image = cv.imread(photo_loc)
    cropped_image = pose_client.cropping_tool.crop_image(image)
    pose_2d_gt_cropped = pose_client.cropping_tool.crop_pose(pose_2d_gt)

    if (pose_client.modes["mode_2d"] == "gt" or pose_client.modes["mode_2d"] == "gt_with_noise"):
        pose_2d = pose_2d_gt_cropped.clone()
        if (pose_client.modes["mode_2d"] == "gt_with_noise"):
            pose_2d = add_noise_to_pose(pose_2d,  pose_client.NOISE_2D_STD, my_rng, noise_type="proj")

        heatmaps = None
        if (pose_client.modes["mode_lift"]=="lift"):
            heatmaps = create_heatmap(pose_2d.numpy().copy(), pose_client.SIZE_X,  pose_client.SIZE_Y)

    elif (pose_client.modes["mode_2d"] == "openpose"):  
        pose_2d, heatmaps, _, _ =  openpose_module.run_only_model(cropped_image, [0.5, 0.75, 1, 1.25, 1.5, 2])

        pose_client.openpose_error = torch.mean(torch.norm(pose_2d_gt_cropped-pose_2d, dim=0))
        if not pose_client.USE_SINGLE_JOINT:
            arm_joints, _, _ = return_arm_joints()
            leg_joints, _, _ = return_leg_joints()
            pose_client.openpose_arm_error = torch.mean(torch.norm(pose_2d_gt_cropped[:, arm_joints]-pose_2d[:, arm_joints], dim=0))
            pose_client.openpose_leg_error = torch.mean(torch.norm(pose_2d_gt_cropped[:, leg_joints]-pose_2d[:, leg_joints], dim=0))
    
    return pose_2d.clone(), pose_2d_gt_cropped.clone(), heatmaps, cropped_image

def find_2d_pose_gt(projection_client, current_state):
    camera_id, bone_pos_3d_GT, _, inv_transformation_matrix, _ = current_state.get_frame_parameters()
    pose_2d_torch = projection_client.take_single_projection(torch.from_numpy(bone_pos_3d_GT).float(), inv_transformation_matrix, camera_id)
    return pose_2d_torch.clone()

def find_lifted_pose(pose_2d, cropped_image, heatmap_2d):
    num_of_joints = pose_2d.shape[1]
    pose = torch.cat((torch.t(pose_2d), torch.ones(num_of_joints,1)), 1)
    pose3d_lift, _, _ = liftnet_module.run(cropped_image, heatmap_2d, pose)
    pose3d_lift = pose3d_lift.view(num_of_joints+2,  -1).permute(1, 0)
    pose3d_lift = rearrange_bones_to_mpi(pose3d_lift, is_torch=True)
    return pose3d_lift

def determine_relative_3d_pose(pose_client, current_state, my_rng, pose_2d, cropped_image, heatmap_2d):
    if not pose_client.USE_LIFT_TERM or pose_client.USE_SINGLE_JOINT or pose_client.is_calibrating_energy:
        return None

    _, current_pose_3d_GT, _, _, transformation_matrix = current_state.get_frame_parameters()
    bone_connections, _, _, hip_index = pose_client.model_settings()

    if (pose_client.modes["mode_lift"] != 'lift'):
        pose3d_relative = torch.from_numpy(current_pose_3d_GT).clone()
        if (pose_client.modes["mode_lift"] == "gt_with_noise"):
            pose3d_relative = add_noise_to_pose(pose3d_relative, pose_client.NOISE_LIFT_STD, my_rng, noise_type="lift")
    elif (pose_client.modes["mode_lift"]   == 'lift'):
        pose3d_lift = find_lifted_pose(pose_2d, cropped_image, heatmap_2d.cpu().numpy())
        pose3d_relative = pose_client.projection_client.camera_to_world(pose3d_lift.cpu(), transformation_matrix)

    if pose_client.LIFT_METHOD == "complex":
        pose3d_lift_directions = calculate_bone_directions(pose3d_relative, np.array(return_lift_bone_connections(bone_connections)), batch=False) 
    if pose_client.LIFT_METHOD == "simple":
        pose3d_lift_directions = calculate_bone_directions_simple(lift_bones=pose3d_relative, bone_lengths=pose_client.boneLengths, 
                                                                bone_length_method=pose_client.BONE_LEN_METHOD, 
                                                                bone_connections=np.array(bone_connections), 
                                                                hip_index=hip_index, batch=False) 

    return pose3d_lift_directions

def initialize_empty_frames(linecount, pose_client, current_state, file_manager, my_rng):
    objective_calib.my_init(pose_client)
    objective_online.my_init(pose_client)

    plot_loc, photo_loc = file_manager.plot_loc, file_manager.get_photo_loc()
    bone_connections, joint_names, num_of_joints, hip_index = pose_client.model_settings()
    camera_id, pose_3d_gt, _, _, transformation_matrix = current_state.get_frame_parameters()

    pose_2d, _, _ = prepare_frames_for_optimization(-1, pose_client, current_state, my_rng, file_manager, init_empty_frames=True)
    file_manager.save_pose_2d(pose_2d, -1)
    superimpose_on_image(pose_2d.cpu().numpy(), plot_loc, -1, bone_connections, photo_loc, custom_name="projected_res_", scale = -1)

    #initial frames
    if pose_client.INIT_POSE_MODE == "gt" or pose_client.INIT_POSE_MODE == "gt_with_noise":
        optimized_poses = pose_3d_gt.copy()
    elif pose_client.INIT_POSE_MODE == "zeros":
        optimized_poses = np.zeros([3,num_of_joints])
    elif pose_client.INIT_POSE_MODE == "backproj" or pose_client.INIT_POSE_MODE == "initial_optimization":
        optimized_poses = pose_client.projection_client.take_single_backprojection(pose_2d, transformation_matrix, joint_names, camera_id)

        if not pose_client.is_calibrating_energy:
            optimized_poses = scale_with_bone_lengths(optimized_poses, pose_client.boneLengths, pose_client.BONE_LEN_METHOD, np.array(bone_connections), batch=False).numpy()
        else:
            optimized_poses = optimized_poses.numpy()
    plot_human(optimized_poses, optimized_poses, plot_loc, -1, bone_connections, pose_client.USE_SINGLE_JOINT, pose_client.animation, 1000, additional_text = 1000)
            
    if not pose_client.is_calibrating_energy:
        optimized_poses = np.repeat(optimized_poses[np.newaxis, :, :], pose_client.ONLINE_WINDOW_SIZE, axis=0)

    if pose_client.INIT_POSE_MODE == "gt_with_noise":
        optimized_poses = add_noise_to_pose(torch.from_numpy(optimized_poses), pose_client.NOISE_3D_INIT_STD, my_rng, noise_type="initial")
        optimized_poses = optimized_poses.numpy()

    if pose_client.INIT_POSE_MODE == "initial_optimization":
        pose_client.pose_3d_preoptimization = optimized_poses.copy()
        optimized_poses, _, _, _ = perform_optimization(pose_client, linecount)

    pose_client.update3dPos(optimized_poses, optimized_poses)
    pose_client.set_initial_pose()

    if pose_client.USE_TRAJECTORY_BASIS:
        raise NotImplementedError

    if pose_client.is_calibrating_energy:
        plot_human(pose_3d_gt,optimized_poses,plot_loc,-1,bone_connections,pose_client.USE_SINGLE_JOINT, pose_client.animation, 1000, additional_text = 1000)
    else:
        plot_all_optimization_results(optimized_poses, pose_client.poses_3d_gt, pose_client.FUTURE_WINDOW_SIZE, plot_loc, -1, bone_connections, pose_client.animation,  pose_client.errors, pose_client.average_errors)


def determine_openpose_error(linecount, pose_client, current_state, plot_loc, photo_loc):
    _, bone_pos_3d_GT, _,  inv_transformation_matrix, _ = current_state.get_frame_parameters()
    bone_connections, _, num_of_joints, _ =  pose_client.model_settings()

    pose_2d_cropped, pose_2d_gt_cropped, heatmap_2d, cropped_image = determine_2d_positions(pose_client=pose_client, current_state=current_state, my_rng=my_rng, photo_loc=photo_loc, linecount=linecount)
    pose3d_lift = determine_relative_3d_pose(pose_client=pose_client, current_state=current_state, pose_2d=pose_2d_cropped, cropped_image=cropped_image, heatmap_2d=heatmap_2d)
    pose_2d = pose_client.cropping_tool.uncrop_pose(pose_2d_cropped)
    pose_2d_gt = pose_client.cropping_tool.uncrop_pose(pose_2d_gt_cropped)

    #pose_client.update3dPos(bone_pos_3d_GT, bone_pos_3d_GT)

    plot_end = {"est": bone_pos_3d_GT, "GT": bone_pos_3d_GT, "drone": current_state.C_drone_gt, "eval_time": 0}
    pose_client.append_res(plot_end)
    return pose_2d, pose3d_lift

def prepare_frames_for_optimization(linecount, pose_client, current_state, my_rng, file_manager, init_empty_frames):
    plot_loc, photo_loc = file_manager.plot_loc, file_manager.get_photo_loc()
    camera_id, current_pose_3d_gt, futuremost_pose_3d_gt, inv_transformation_matrix, transformation_matrix = current_state.get_frame_parameters()

    #find 2d pose (using openpose or gt)
    pose_2d_cropped, pose_2d_gt_cropped, heatmap_2d, cropped_image = determine_2d_positions(pose_client=pose_client, current_state=current_state, my_rng=my_rng, file_manager=file_manager, linecount=linecount)
    
    #find relative 3d pose using liftnet or GT relative pose
    pose3d_lift_directions = determine_relative_3d_pose(pose_client=pose_client, current_state=current_state, my_rng=my_rng, pose_2d=pose_2d_cropped, cropped_image=cropped_image, heatmap_2d=heatmap_2d)
        
    #uncrop 2d pose
    pose_2d = pose_client.cropping_tool.uncrop_pose(pose_2d_cropped)
    pose_2d_gt = pose_client.cropping_tool.uncrop_pose(pose_2d_gt_cropped)

    #add information you need to your window
    if init_empty_frames:
        future_poses_3d_gt = current_state.get_first_future_poses()
        pose_client.init_frames(pose_2d, pose_2d_gt, inv_transformation_matrix, pose3d_lift_directions, current_pose_3d_gt, future_poses_3d_gt, camera_id)
    else:
        pose_client.addNewFrame(linecount, pose_2d, pose_2d_gt, inv_transformation_matrix, pose3d_lift_directions, current_pose_3d_gt, futuremost_pose_3d_gt, camera_id)
  
    return pose_2d, pose3d_lift_directions, cropped_image


def perform_optimization(pose_client, linecount):
    result_shape, result_size, loss_dict = pose_client.result_shape, pose_client.result_size, pose_client.loss_dict

    pose3d_init = np.reshape(a = pose_client.pose_3d_preoptimization.copy(), newshape = [result_size,], order = "C")
    assert not np.isnan(pose3d_init).any()

    if pose_client.USE_TRAJECTORY_BASIS:
        raise NotImplementedError

    if (pose_client.is_calibrating_energy): 
        objective = objective_calib
        objective_jacobian =  objective_calib.jacobian
    else:
        objective = objective_online
        objective_jacobian = objective_online.jacobian

    bounds = (-np.inf, np.inf)#(np.min(pose3d_init)-20, np.max(pose3d_init)+20)

    start_time = time.time()
    objective.reset_current(pose_client)
    optimized_res = least_squares(objective.forward, pose3d_init, jac=objective_jacobian, bounds=bounds, method=pose_client.method, ftol=pose_client.ftol, xtol=pose_client.xtol)
    optimization_losses_weighted = objective.pltpts_weighted
    optimization_losses = objective.pltpts
    if not pose_client.is_calibrating_energy:
        objective.reset_future(pose_client)
        optimized_res = least_squares(objective.forward, optimized_res.x, jac=objective_jacobian, bounds=bounds, method=pose_client.method, ftol=pose_client.ftol, xtol=pose_client.xtol)

    func_eval_time = time.time() - start_time
    print("least squares eval time", func_eval_time)
    if not pose_client.USE_TRAJECTORY_BASIS:
        optimized_poses = np.reshape(a = optimized_res.x, newshape = result_shape, order = "C")
    else:
        optimized_traj = np.reshape(a = optimized_res.x, newshape = result_shape, order = "C")
        optimized_poses = project_trajectory(torch.from_numpy(optimized_traj).float(), pose_client.ONLINE_WINDOW_SIZE, pose_client.NUMBER_OF_TRAJ_PARAM).numpy()
        pose_client.optimized_traj = optimized_traj

    #adjusted_optimized_poses = adjust_with_M(pose_client.M, optimized_poses, hip_index)
    adjusted_optimized_poses = optimized_poses.copy()

    return optimized_poses, adjusted_optimized_poses, optimization_losses_weighted, func_eval_time


def determine_positions(linecount, pose_client, current_state, file_manager, my_rng):
    plot_loc, photo_loc = file_manager.plot_loc, file_manager.get_photo_loc()
    bone_connections, joint_names, num_of_joints, hip_index = pose_client.model_settings()
    camera_index, current_pose_3d_gt, futuremost_pose_3d_gt, inv_transformation_matrix, transformation_matrix = current_state.get_frame_parameters()

    pose_2d, pose3d_lift_directions, cropped_image =  prepare_frames_for_optimization(linecount, pose_client, current_state, my_rng, file_manager, init_empty_frames=False)
    file_manager.save_pose_2d(pose_2d, linecount)
    file_manager.save_lift(pose3d_lift_directions, linecount)

    #add current pose as initial pose
    pose_client.set_initial_pose()

    optimized_poses, adjusted_optimized_poses, optimization_losses, func_eval_time = perform_optimization(pose_client, linecount)
    pose_client.update3dPos(optimized_poses, adjusted_optimized_poses)
    if (pose_client.is_calibrating_energy):
        pose_client.update_bone_lengths(torch.from_numpy(optimized_poses).float())

    #lots of plot stuff
    errors = pose_client.calculate_store_errors(linecount)

    if (plot_loc != 0 and not pose_client.quiet): 
        start_plot_time = time.time()
        check = pose_client.projection_client.take_single_projection(torch.from_numpy(pose_client.current_pose).float(), inv_transformation_matrix, camera_index)
        superimpose_on_image(pose_2d.cpu().numpy(), plot_loc, linecount, bone_connections, photo_loc, custom_name="projected_res_", scale = -1, projection=check.cpu().numpy())
        #superimpose_on_image(pose_2d.numpy(), plot_loc, linecount, bone_connections, photo_loc, custom_name="projected_res_2_", scale = -1)
        #plot_2d_projection(check.numpy(), plot_loc, linecount, bone_connections, custom_name="proj_2d")
        if pose_client.modes["mode_2d"] == "openpose":
            display_image(cropped_image, plot_loc, linecount)
        plot_human(current_pose_3d_gt, pose_client.adj_current_pose, plot_loc, linecount, bone_connections, pose_client.USE_SINGLE_JOINT, pose_client.animation, 1000, additional_text = 1000)
        #plot_human(current_pose_3d_gt, noisy_init_pose, plot_loc, linecount, bone_connections, 0, custom_name="init_pose", label_names = ["GT", "Init"])
        #save_heatmaps(heatmap_2d, linecount, plot_loc)
        #save_heatmaps(heatmaps_scales.cpu().numpy(), client.linecount, plot_loc, custom_name = "heatmaps_scales_", scales=scales, poses=poses_scales.cpu().numpy(), bone_connections=bone_connections)
        plot_optimization_losses(optimization_losses, plot_loc, linecount, pose_client.loss_dict)

        if (not pose_client.is_calibrating_energy and not pose_client.simulate_error_mode):
            plot_future_poses(adjusted_optimized_poses, pose_client.FUTURE_WINDOW_SIZE, plot_loc, linecount, bone_connections, pose_client.animation)
            plot_all_optimization_results(adjusted_optimized_poses, pose_client.poses_3d_gt, pose_client.FUTURE_WINDOW_SIZE, plot_loc, linecount, bone_connections, pose_client.animation,  pose_client.errors, pose_client.average_errors)
            #plot_human(current_pose_3d_gt, pose_client.adj_current_pose, plot_loc, linecount-pose_client.MIDDLE_POSE_INDEX+1, bone_connections, pose_client.USE_SINGLE_JOINT, pose_client.animation, errors["middle_error"], custom_name="middle_pose_", label_names = ["GT", "Estimate"], additional_text = errors["ave_middle_error"])
            if pose_client.modes["mode_lift"] == "lift":
                hip_joint_gt = current_pose_3d_gt[:,hip_index]
                plot_human(current_pose_3d_gt-hip_joint_gt[:, np.newaxis], pose3d_lift_directions.numpy(), plot_loc, linecount, bone_connections, pose_client.USE_SINGLE_JOINT, pose_client.animation, -1, custom_name="lift_res_", label_names = ["LiftNet", "GT"])
        end_plot_time = time.time()
        print("Time it took to plot", end_plot_time - start_plot_time)
    plot_end = {"est": pose_client.adj_current_pose, "GT": current_pose_3d_gt, "drone": current_state.C_drone_gt, "eval_time": func_eval_time}
    pose_client.append_res(plot_end)
    #file_manager.write_reconstruction_values(adjusted_current_pose, current_pose_3d_gt, current_state.C_drone_gt, current_state.R_drone_gt, linecount, num_of_joints)
    
    current_state.update_human_info(pose_client.current_pose)
    file_manager.record_reconstruction_values(pose_client.current_pose, linecount)
    return  pose_client.adj_current_pose

def switch_energy(value):
    pass
