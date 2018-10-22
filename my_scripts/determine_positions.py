from helpers import * 
from State import *
from NonAirSimClient import *
from PoseEstimationClient import *
from pose3d_optimizer import *
from pose3d_optimizer_scipy import *
from project_bones import *
import numpy as np
import torch
import cv2 as cv
from torch.autograd import Variable
import time
from scipy.optimize import least_squares

import util as demo_util

import openpose as openpose_module
import liftnet as liftnet_module

objective_flight = pose3d_flight_scipy()
objective_calib = pose3d_calibration_scipy()
objective_future = pose3d_future()

CURRENT_POSE_INDEX = 1
FUTURE_POSE_INDEX = 0

def determine_all_positions(airsim_client, pose_client,  plot_loc = 0, photo_loc = 0):
    airsim_client.simPause(True)
    if (pose_client.modes["mode_3d"] == 0):
        positions, unreal_positions = determine_3d_positions_all_GT(airsim_client, pose_client, plot_loc, photo_loc)
    elif (pose_client.modes["mode_3d"] == 1):
        positions, unreal_positions = determine_3d_positions_backprojection(airsim_client, pose_client, plot_loc, photo_loc)
    elif (pose_client.modes["mode_3d"] == 2):            
        positions, unreal_positions = determine_3d_positions_energy_pytorch(airsim_client, pose_client, plot_loc, photo_loc)
    elif (pose_client.modes["mode_3d"] == 3):
        positions, unreal_positions = determine_3d_positions_energy_scipy(airsim_client, pose_client, plot_loc, photo_loc)
    airsim_client.simPause(False)

    return positions, unreal_positions

def determine_2d_positions(mode_2d, cropping_tool,  return_heatmaps=True, is_torch = True, unreal_positions = 0, bone_pos_3d_GT = 0, input_image = 0,  scales = [1]):
    if (mode_2d == 0):
        bone_2d, heatmaps = find_2d_pose_gt(unreal_positions, bone_pos_3d_GT, input_image, cropping_tool, return_heatmaps, is_torch)
        heatmaps_scales = 0
        poses_scales = 0
    elif (mode_2d == 1):            
        bone_2d, heatmaps, heatmaps_scales, poses_scales = find_2d_pose_openpose(input_image,  scales)
    return bone_2d, heatmaps, heatmaps_scales, poses_scales

def find_2d_pose_gt(unreal_positions, bone_pos_3d_GT, input_image, cropping_tool, return_heatmaps=True, is_torch = True):
    if (is_torch):
        R_drone_unreal = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False) #pitch roll yaw
        C_drone_unreal = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)
        bone_pos_GT = Variable(torch.from_numpy(bone_pos_3d_GT).float(), requires_grad = True)
        R_cam_np = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, pi/2, CAMERA_YAW_OFFSET, returnTensor = False)
        R_cam = Variable(torch.from_numpy(R_cam_np).float(), requires_grad = False)
        bone_2d_var, heatmaps = take_bone_projection_pytorch(bone_pos_GT, R_drone_unreal.cpu(), C_drone_unreal.cpu(), R_cam.cpu())
        bone_2d = bone_2d_var.detach()
        if (return_heatmaps):
            bone_2d = cropping_tool.crop_pose(bone_2d)
            heatmaps = create_heatmap(bone_2d.data.cpu().numpy(), input_image.shape[1], input_image.shape[0])
        else:
            heatmaps = 0
    else:
        R_drone_unreal = euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
        C_drone_unreal = unreal_positions[DRONE_POS_IND, :]
        C_drone_unreal = C_drone_unreal[:, np.newaxis]
        bone_2d, heatmaps = take_bone_projection(bone_pos_3d_GT, R_drone_unreal, C_drone_unreal)
        if (return_heatmaps):
            bone_2d = cropping_tool.crop_pose(bone_2d)
            heatmaps = create_heatmap(bone_2d, input_image.shape[1], input_image.shape[0])
        else:
            heatmaps = 0
    return bone_2d, heatmaps

def find_2d_pose_openpose(input_image, scales):
    poses, heatmaps, heatmaps_scales, poses_scales = openpose_module.run_only_model(input_image, scales)
    return poses, heatmaps.cpu().numpy(), heatmaps_scales, poses_scales

def determine_relative_3d_pose(mode_lift, bone_2d, cropped_image, heatmap_2d, R_drone, C_drone, bone_pos_3d_GT):
    if (mode_lift == 0):
        pose3d_relative = bone_pos_3d_GT

    elif (mode_lift == 1):
        num_of_joints = bone_2d.shape[1]
        pose = torch.cat((torch.t(bone_2d), torch.ones(num_of_joints,1)), 1)
        pose3d_lift, _, _ = liftnet_module.run(cropped_image, heatmap_2d, pose)
        pose3d_lift = pose3d_lift.view(num_of_joints+2,  -1).permute(1, 0)
        pose3d_lift = rearrange_bones_to_mpi(pose3d_lift, True)
        pose3d_relative = camera_to_world(R_drone, C_drone, pose3d_lift.cpu().data.numpy(), is_torch = False)
    return pose3d_relative


def determine_3d_positions_energy_scipy(airsim_client, pose_client, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = airsim_client.getSynchronizedData()
    bone_connections, joint_names, num_of_joints, bone_pos_3d_GT = model_settings(pose_client.model, bone_pos_3d_GT)
    input_image = cv.imread(photo_loc)
    cropped_image, scales = pose_client.cropping_tool.crop(input_image, airsim_client.linecount)

    R_drone = euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
    C_drone = unreal_positions[DRONE_POS_IND, :]
    C_drone = C_drone[:, np.newaxis]

    #find 2d pose (using openpose or gt)
    bone_2d, heatmap_2d, _, _ = determine_2d_positions(pose_client.modes["mode_2d"], pose_client.cropping_tool, True, True, unreal_positions, bone_pos_3d_GT, cropped_image, scales)

    #find relative 3d pose using liftnet or GT relative pose
    pose3d_lift = determine_relative_3d_pose(pose_client.modes["mode_lift"], bone_2d, cropped_image, heatmap_2d, R_drone, C_drone, bone_pos_3d_GT)
    bone_2d = bone_2d.cpu().numpy() 
        
    #find liftnet bone directions and save them
    lift_bone_directions = return_lift_bone_connections(bone_connections)
    pose3d_lift_directions = np.zeros([3, len(lift_bone_directions)])
    for i, bone in enumerate(lift_bone_directions):
        bone_vector = pose3d_lift[:, bone[0]] - pose3d_lift[:, bone[1]]
        pose3d_lift_directions[:, i] = bone_vector/(np.linalg.norm(bone_vector)+EPSILON)

    #uncrop 2d pose     
    bone_2d = pose_client.cropping_tool.uncrop_pose(bone_2d)

    #add current pose as initial pose. if first frame, take backprojection for initialization
    if (airsim_client.linecount != 0):
        pre_pose_3d = pose_client.future_pose#poseList_3d[0]
    else:
        pre_pose_3d = take_bone_backprojection(bone_2d, R_drone, C_drone, joint_names)
        pose_client.poseList_3d_calibration = pre_pose_3d
       # pose_client.kalman.init_state(np.reshape(a = pre_pose_3d, newshape = [3*num_of_joints,1], order = "F"))

    #add information you need to your window
    if (pose_client.isCalibratingEnergy): 
        pose_client.addNewCalibrationFrame(bone_2d, R_drone, C_drone, pre_pose_3d)
    pose_client.addNewFrame(bone_2d, R_drone, C_drone, pre_pose_3d, pose3d_lift_directions)

    final_loss = np.zeros([1,1])

    if (airsim_client.linecount > 0):
        #calibration mode parameters
        if (pose_client.isCalibratingEnergy): 
            loss_dict = pose_client.loss_dict_calib
            objective = objective_calib
            pose3d_init_scrambled = pose_client.poseList_3d_calibration.copy()
            result_shape = pose_client.result_shape_calib
            result_size = result_shape[0]*result_shape[1]
            pose3d_init = np.reshape(a = pose3d_init_scrambled, newshape = [result_size,], order = "C")
            objective.reset(pose_client)
            objective_jacobian =  objective_calib.jacobian

        #flight mode parameters
        else:
            loss_dict = pose_client.loss_dict_flight
            result_shape = pose_client.result_shape_flight
            result_size = result_shape[0]*result_shape[1]*result_shape[2]
            pose3d_init = np.zeros(result_shape)
            for queue_index, pose3d_ in enumerate(pose_client.poseList_3d):
                pose3d_init[queue_index+1, :] = pose3d_.copy()
            pose3d_init[FUTURE_POSE_INDEX, :] = pose3d_init[CURRENT_POSE_INDEX, :] #initialize future pose as current pose
            pose3d_init = np.reshape(a = pose3d_init, newshape = [result_size,], order = "C")
            objective = objective_flight
            objective.reset(pose_client)
            objective_jacobian = objective_flight.jacobian

        start_time = time.time()
        optimized_res = least_squares(objective.forward, pose3d_init, jac=objective_jacobian, bounds=(-np.inf, np.inf), method=pose_client.method, ftol=pose_client.ftol)
        func_eval_time = time.time() - start_time
        print("least squares eval time", func_eval_time)
        P_world_scrambled = optimized_res.x
        P_world = np.reshape(a = P_world_scrambled, newshape = result_shape, order = "C")

        if (pose_client.isCalibratingEnergy):
            optimized_3d_pose = P_world
            pose_client.update3dPos(optimized_3d_pose, is_calib = pose_client.isCalibratingEnergy)
            pose_client.future_pose = optimized_3d_pose
            if airsim_client.linecount > 3:
                for i, bone in enumerate(bone_connections):
                    pose_client.boneLengths[i] = np.sum(np.square(optimized_3d_pose[:, bone[0]] - optimized_3d_pose[:, bone[1]]))
        else:
            optimized_3d_pose =  P_world[CURRENT_POSE_INDEX, :,:] #current pose
            pose_client.future_pose =  P_world[FUTURE_POSE_INDEX, :,:] #future pose
            pose_client.update3dPos(P_world, is_calib = pose_client.isCalibratingEnergy)

        if (pose_client.calc_hess):
            
            #start_find_hess = time.time()
            #hess = objective.mini_hessian(P_world)
            #hess = objective.hessian(P_world)
            #hess = objective.approx_hessian(P_world)
            #end_find_hess = time.time()
            #inv_hess = np.linalg.inv(hess)
            #pose_client.updateMeasurementCov(inv_hess, CURRENT_POSE_INDEX, FUTURE_POSE_INDEX)
            #pose_client.updateMeasurementCov_mini(inv_hess, CURRENT_POSE_INDEX, FUTURE_POSE_INDEX)
            #print("Time for finding hessian:", end_find_hess-start_find_hess)

            #find candidate drone positions
            potential_states_fetcher = Potential_States_Fetcher(C_drone, optimized_3d_pose, pose_client.future_pose, pose_client.model)
            potential_states = potential_states_fetcher.get_potential_positions_spherical()

            if not pose_client.isCalibratingEnergy:
                #find hessian for each drone position
                potential_states_fetcher.find_hessians_for_potential_states(objective_future, pose_client, P_world)
                goal_state = potential_states_fetcher.find_best_potential_state()
                pose_client.goal_state = goal_state
                pose_client.goal_indices.append(potential_states_fetcher.goal_state_ind)

                if not pose_client.quiet:
                    #plot potential states and current state, projections of each state, cov's of each state
                    plot_potential_states(optimized_3d_pose, pose_client.future_pose, bone_pos_3d_GT, potential_states, C_drone, R_drone, pose_client.model, plot_loc, airsim_client.linecount)
                    #plot_potential_ellipses(optimized_3d_pose, pose_client.future_pose, bone_pos_3d_GT, potential_states, None, pose_client.model, plot_loc, airsim_client.linecount, ellipses=False)
                    plot_potential_projections(potential_states_fetcher.potential_pose2d_list, airsim_client.linecount, plot_loc, photo_loc, pose_client.model)

                    plot_potential_hessians(potential_states_fetcher.potential_covs_normal, airsim_client.linecount, plot_loc, custom_name = "potential_covs_normal_")
                    #plot_potential_hessians(potential_hessians_normal, airsim_client.linecount, plot_loc, custom_name = "potential_hess_normal_")
                    plot_potential_ellipses(optimized_3d_pose, pose_client.future_pose, bone_pos_3d_GT, potential_states_fetcher, pose_client.model, plot_loc, airsim_client.linecount)

    #if the frame is the first frame, the energy is found through backprojection
    else:
        optimized_3d_pose = pre_pose_3d
        loss_dict = CALIBRATION_LOSSES
        func_eval_time = 0
        pose_client.future_pose = optimized_3d_pose
        
    pose_client.error_2d.append(final_loss[0])

    root_est = optimized_3d_pose[:,joint_names.index('spine1')]
    optimized_3d_pose = np.dot(optimized_3d_pose - root_est[:, np.newaxis], pose_client.M)+root_est[:, np.newaxis]
    root_future = pose_client.future_pose[:,joint_names.index('spine1')]
    M_future_pose = np.dot(pose_client.future_pose - root_future[:, np.newaxis], pose_client.M)+root_future[:, np.newaxis]

    #check,  _ = take_bone_projection(optimized_3d_pose, R_drone, C_drone)

    #lots of plot stuff
    error_3d = np.mean(np.linalg.norm(bone_pos_3d_GT - optimized_3d_pose, axis=0))
    pose_client.error_3d.append(error_3d)
    if (plot_loc != 0 and not pose_client.quiet): 
        #superimpose_on_image(bone_2d, plot_loc, airsim_client.linecount, bone_connections, photo_loc, custom_name="projected_res_", scale = -1, projection=check)
        plot_human(bone_pos_3d_GT, optimized_3d_pose, plot_loc, airsim_client.linecount, bone_connections, error_3d)
        #save_heatmaps(heatmap_2d, airsim_client.linecount, plot_loc)
        #save_heatmaps(heatmaps_scales.cpu().numpy(), client.linecount, plot_loc, custom_name = "heatmaps_scales_", scales=scales, poses=poses_scales.cpu().numpy(), bone_connections=bone_connections)

        if (not pose_client.isCalibratingEnergy):
            #plot_human(optimized_3d_pose, M_future_pose, plot_loc, airsim_client.linecount, bone_connections, error_3d, custom_name="future_plot_", label_names = ["current", "future"])
            pose3d_lift_normalized, _ = normalize_pose(pose3d_lift, joint_names, is_torch=False)
            bone_pos_3d_GT_normalized, _ = normalize_pose(bone_pos_3d_GT, joint_names, is_torch=False)
            optimized_3d_pose_normalized, _ = normalize_pose(optimized_3d_pose, joint_names, is_torch=False)
            plot_human(bone_pos_3d_GT_normalized, pose3d_lift_normalized, plot_loc, airsim_client.linecount, bone_connections, error_3d, custom_name="lift_res_", label_names = ["GT", "LiftNet"])
            #plot_human(pose3d_lift_normalized, optimized_3d_pose_normalized, plot_loc, airsim_client.linecount, bone_connections, error_3d, custom_name="lift_res_2_", label_names = ["LiftNet", "Estimate"])
            #plot_optimization_losses(objective.pltpts, plot_loc, airsim_client.linecount, loss_dict)

    positions = form_positions_dict(angle, drone_pos_vec, optimized_3d_pose[:,joint_names.index('spine1')])
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    plot_end = {"est": optimized_3d_pose, "GT": bone_pos_3d_GT, "drone": C_drone, "eval_time": func_eval_time, "f_string": f_output_str}
    pose_client.append_res(plot_end)

    return positions, unreal_positions


def determine_3d_positions_energy_pytorch(airsim_client, pose_client, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = airsim_client.getSynchronizedData()
    bone_connections, joint_names, num_of_joints, bone_pos_3d_GT = model_settings(pose_client.model, bone_pos_3d_GT)
    input_image = cv.imread(photo_loc)
    cropped_image = pose_client.cropping_tool.crop(input_image)
    scales = pose_client.cropping_tool.scales

    #DONT FORGET THESE CHANGES
    R_drone = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False)
    C_drone = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)

    #find 2d pose (using openpose or gt)
    bone_2d, heatmap_2d, _, _ = determine_2d_positions(pose_client.modes["mode_2d"], pose_client.cropping_tool, True, True, unreal_positions, bone_pos_3d_GT, cropped_image, scales)

    #find 3d pose using liftnet
    pose = torch.cat((torch.t(bone_2d), torch.ones(num_of_joints,1)), 1)
    input_image = cv.imread(photo_loc)
    pose3d_lift, _, _ = liftnet_module.run(input_image, heatmap_2d.cpu().numpy(), pose)
    pose3d_lift = pose3d_lift.view(num_of_joints+2,  -1).permute(1, 0)
    pose3d_lift = rearrange_bones_to_mpi(pose3d_lift, True)
    pose3d_lift = camera_to_world(R_drone, C_drone, pose3d_lift.cpu(), is_torch = True)

    #find lift bone directions
    pose3d_lift_directions = torch.zeros([3, num_of_joints-1])
    for i, bone in enumerate(bone_connections):
        bone_vector = pose3d_lift[:, bone[0]] - pose3d_lift[:, bone[1]]
        pose3d_lift_directions[:, i] = bone_vector/(torch.norm(bone_vector)+EPSILON)

    if (airsim_client.linecount != 0):
        pose3d_ = pose_client.poseList_3d[-1]
    else:
        pose3d_ = take_bone_backprojection_pytorch(bone_2d, R_drone, C_drone, joint_names)

    if (pose_client.isCalibratingEnergy): 
        pose_client.addNewCalibrationFrame(bone_2d, R_drone, C_drone, pose3d_)
    pose_client.addNewFrame(bone_2d, R_drone, C_drone, pose3d_, pose3d_lift_directions)

    pltpts = {}
    final_loss = np.zeros([1,1])
    if (airsim_client.linecount >1):
        #calibration mode parameters
        if (pose_client.isCalibratingEnergy): 
            objective = pose3d_calibration(pose_client.model)
            optimizer = torch.optim.SGD(objective.parameters(), lr = 0.0005, momentum=0.9)
            #optimizer = torch.optim.LBFGS(objective.parameters(), lr = 0.0005, max_iter=50)
            if (airsim_client.linecount < 8):
                num_iterations = 5000
            else:
                num_iterations = 500
            objective.init_pose3d(pose3d_) 
            loss_dict = pose_client.loss_dict_calib
            data_list = pose_client.requiredEstimationData_calibration
            energy_weights = pose_client.weights_calib
         #flight mode parameters
        else:
            objective = pose3d_flight(pose_client.boneLengths, pose_client.FLIGHT_WINDOW_SIZE, pose_client.model)
            optimizer = torch.optim.SGD(objective.parameters(), lr =1, momentum=0.8)
            #optimizer = torch.optim.LBFGS(objective.parameters(), lr = 0.0005,  max_iter=50)
            num_iterations = 5000
            #init all 3d pose 
            for queue_index, pose3d_ in enumerate(pose_client.poseList_3d):

                objective.init_pose3d(pose3d_, queue_index)
            loss_dict = pose_client.loss_dict_flight
            data_list = pose_client.requiredEstimationData
            energy_weights = pose_client.weights_flight

        for loss_key in loss_dict:
            pltpts[loss_key] = np.zeros([num_iterations])

        start_time = time.time()
        for i in range(num_iterations):
            def closure():
                outputs = {}
                output = {}
                for loss_key in loss_dict:
                    outputs[loss_key] = []
                    output[loss_key] = 0
                optimizer.zero_grad()
                objective.zero_grad()

                queue_index = 0
                for bone_2d_, R_drone_, C_drone_ in data_list:
                    if (pose_client.isCalibratingEnergy):
                        loss = objective.forward(bone_2d_, R_drone_, C_drone_)
                    else: 
                        pose3d_lift_directions = pose_client.liftPoseList[queue_index]
                        loss = objective.forward(bone_2d_, R_drone_, C_drone_, pose3d_lift_directions, queue_index)

                        if (i == num_iterations - 1 and queue_index == 0):
                            normalized_pose_3d, _ = normalize_pose(objective.pose3d[0, :, :], joint_names, is_torch = True)
                            normalized_lift, _ = normalize_pose(pose3d_lift, joint_names, is_torch = True)
                            bone_pos_3d_GT
                            plot_human(normalized_pose_3d.data.numpy(), normalized_lift.cpu().numpy(), plot_loc, airsim_client.linecount, bone_connections, custom_name="lift_res_", orientation = "z_up")
                    for loss_key in loss_dict:
                        outputs[loss_key].append(loss[loss_key])
                    queue_index += 1

                overall_output = Variable(torch.FloatTensor([0]))
                for loss_key in loss_dict:
                    output[loss_key] = (sum(outputs[loss_key])/len(outputs[loss_key]))
                    overall_output += energy_weights[loss_key]*output[loss_key]
                    pltpts[loss_key][i] = output[loss_key].data.numpy() 
                    if (i == num_iterations - 1):
                        final_loss[0] += energy_weights[loss_key]*np.copy(output[loss_key].data.numpy())

                overall_output.backward(retain_graph = True)
                return overall_output
            optimizer.step(closure)

        func_eval_time = time.time() - start_time
        print("pytorch eval time", func_eval_time)

        if (pose_client.isCalibratingEnergy):
            P_world = objective.pose3d
            pose_client.update3dPos(P_world, all = True)
            if airsim_client.linecount > 3:
                for i, bone in enumerate(bone_connections):
                    pose_client.boneLengths[i] = torch.sum(torch.pow(P_world[:, bone[0]] - P_world[:, bone[1]],2)).data 
                update_torso_size(0.86*(torch.sqrt(torch.sum(torch.pow(P_world[:, joint_names.index('neck')].data- P_world[:, joint_names.index('spine1')].data, 2)))))   
        else:
            P_world = objective.pose3d[0, :, :]
            pose_client.update3dPos(P_world)

    #if the frame is the first frame, the energy is found through backprojection
    else:
        P_world = pose3d_
        loss_dict = pose_client.loss_dict_calib
        func_eval_time = 0
    
    pose_client.error_2d.append(final_loss[0])
    #check,  _ = take_bone_projection_pytorch(P_world, R_drone, C_drone)

    P_world = P_world.cpu().data.numpy()
    error_3d = np.mean(np.linalg.norm(bone_pos_3d_GT - P_world, axis=0))
    pose_client.error_3d.append(error_3d)
    if (plot_loc != 0):
        #superimpose_on_image(bone_2d.data.numpy(), plot_loc, client.linecount, bone_connections, photo_loc, custom_name="projected_res_", scale = -1, projection=check.data.numpy())
        plot_human(bone_pos_3d_GT, P_world, plot_loc, airsim_client.linecount, bone_connections, error_3d)
        if (airsim_client.linecount >1):
            plot_optimization_losses(pltpts, plot_loc, airsim_client.linecount, loss_dict)

    positions = form_positions_dict(angle, drone_pos_vec, P_world[:,0])
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    plot_end = {"est": P_world, "GT": bone_pos_3d_GT, "drone": C_drone, "eval_time": func_eval_time, "f_string": f_output_str}
    pose_client.append_res(plot_end)

    return positions, unreal_positions

def determine_3d_positions_backprojection(airsim_client, pose_client, plot_loc = 0, photo_loc = 0):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = airsim_client.getSynchronizedData()
    bone_connections, joint_names, _, bone_pos_3d_GT = model_settings(pose_client.model, bone_pos_3d_GT)

    bone_2d, _, _, _ = determine_2d_positions(pose_client.modes["mode_2d"], pose_client.cropping_tool, False, False, unreal_positions, bone_pos_3d_GT, photo_loc, -1)

    R_drone = euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2])
    C_drone = unreal_positions[DRONE_POS_IND, :]
    C_drone = C_drone[:, np.newaxis]
    
    P_world = take_bone_backprojection(bone_2d, R_drone, C_drone, joint_names)
    error_3d = np.linalg.norm(bone_pos_3d_GT - P_world)
    pose_client.error_3d.append(error_3d)

    if (plot_loc != 0):
        check, _, _ = take_bone_projection(P_world, R_drone, C_drone)
        superimpose_on_image([check], plot_loc, airsim_client.linecount, bone_connections, photo_loc)
        plot_human(bone_pos_3d_GT, P_world, plot_loc, airsim_client.linecount, bone_connections, error_3d)

    positions = form_positions_dict(angle, drone_pos_vec, P_world[:,0])
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)
    plot_end = {"est": P_world, "GT": bone_pos_3d_GT, "drone": C_drone, "eval_time": 0, "f_string": f_output_str}
    pose_client.append_res(plot_end)

    return positions, unreal_positions

def determine_3d_positions_all_GT(airsim_client, pose_client, plot_loc, photo_loc):
    unreal_positions, bone_pos_3d_GT, drone_pos_vec, angle = airsim_client.getSynchronizedData()
    bone_connections, joint_names, num_of_joints, bone_pos_3d_GT = model_settings(pose_client.model, bone_pos_3d_GT)
    R_drone = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False)
    C_drone = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)
    
    if (pose_client.modes["mode_2d"] == 1):
        input_image = cv.imread(photo_loc) 

        cropped_image, scales = pose_client.cropping_tool.crop(input_image, airsim_client.linecount)
               
        #find 2d pose (using openpose or gt)
        bone_2d, heatmap_2d, _, _ = determine_2d_positions(pose_client.modes["mode_2d"], pose_client.cropping_tool, True, True, unreal_positions, bone_pos_3d_GT, cropped_image, scales)

        #uncrop 2d pose 
        bone_2d = pose_client.cropping_tool.uncrop_pose(bone_2d)

        if (not pose_client.quiet):
            superimpose_on_image(bone_2d.cpu().numpy(), plot_loc, airsim_client.linecount, bone_connections, photo_loc, custom_name="gt_")
            save_image(cropped_image, airsim_client.linecount, plot_loc, custom_name="cropped_img_")
            save_heatmaps(heatmap_2d, airsim_client.linecount, plot_loc)
            #save_heatmaps(heatmaps_scales.cpu().numpy(), airsim_client.linecount, plot_loc, custom_name = "heatmaps_scales_", scales=scales, poses=poses_scales.cpu().numpy(), bone_connections=bone_connections)


    elif (pose_client.modes["mode_2d"] == 0):
        bone_2d, heatmap_2d, _, _ = determine_2d_positions(pose_client.modes["mode_2d"], pose_client.cropping_tool, False, False, unreal_positions, bone_pos_3d_GT, photo_loc, 0)
        if not pose_client.quiet:
            superimpose_on_image(bone_2d, plot_loc, airsim_client.linecount, bone_connections, photo_loc, custom_name="gt_", scale = 0)

    positions = form_positions_dict(angle, drone_pos_vec, unreal_positions[HUMAN_POS_IND,:])
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]
    positions[R_SHOULDER_IND,:] = unreal_positions[R_SHOULDER_IND,:]
    positions[L_SHOULDER_IND,:] = unreal_positions[L_SHOULDER_IND,:]
    f_output_str = '\t'+str(unreal_positions[HUMAN_POS_IND, 0]) +'\t'+str(unreal_positions[HUMAN_POS_IND, 1])+'\t'+str(unreal_positions[HUMAN_POS_IND, 2])+'\t'+str(angle[0])+'\t'+str(angle[1])+'\t'+str(angle[2])+'\t'+str(drone_pos_vec.x_val)+'\t'+str(drone_pos_vec.y_val)+'\t'+str(drone_pos_vec.z_val)+'\n'
    plot_end = {"est": bone_pos_3d_GT, "GT": bone_pos_3d_GT, "drone": C_drone.cpu().numpy(), "eval_time": 0, "f_string": f_output_str}
    pose_client.append_res(plot_end)
    print(bone_pos_3d_GT[:, joint_names.index("spine1")])
    return positions, unreal_positions

def form_positions_dict(angle, drone_pos_vec, human_pos):
    positions = np.zeros([5, 3])
    positions[DRONE_POS_IND,:] = np.array([drone_pos_vec.x_val, drone_pos_vec.y_val, drone_pos_vec.z_val])
    positions[DRONE_ORIENTATION_IND,:] = np.array([angle[0], angle[1], angle[2]])
    positions[HUMAN_POS_IND,:] = human_pos
    positions[HUMAN_POS_IND,2] = positions[HUMAN_POS_IND,2]
    return positions

def switch_energy(value):
    pass
