import numpy as np
import torch 
import time, os
import airsim
from helpers import vector3r_arr_to_dict, rearrange_bones_to_mpi

def get_client_gt_values(airsim_client, pose_client, simulated_value_dict):
    """
    Description: 
        Sort out values read from simulator and return them

    Inputs: 
        airsim_client: an object of type VehicleClient or DroneFlightClient
        pose_client: an object of type PoseEstimationClient
        simulated_value_dict: dict type input of values read from AirSim simulator (or previously saved values)
    Returns:
        bone_pos_gt: numpy array of shape (3, num_of_joints)
        drone_orientation_gt: numpy array of shape (3,)
        drone_pos_gt: numpy array of shape (3,1)
    """

    drone_orientation_gt = np.array([simulated_value_dict['droneOrient'].x_val, simulated_value_dict['droneOrient'].y_val, 
                            simulated_value_dict['droneOrient'].z_val])

    #convert Unreal engine coordinates to AirSim coordinates.
    bone_pos_gt = np.zeros([3, 21])
    bone_ind = 0
    for ind, attribute in enumerate(airsim.attributes):
        if (attribute != 'dronePos' and attribute != 'droneOrient' and attribute != 'humanPos'):
            bone_pos_gt[:, bone_ind] = np.array([simulated_value_dict[attribute].x_val, simulated_value_dict[attribute].y_val, 
                                        -simulated_value_dict[attribute].z_val]) - airsim_client.DRONE_INITIAL_POS
            bone_pos_gt[:, bone_ind] = bone_pos_gt[:, bone_ind]/100
            bone_ind += 1

    if pose_client.USE_SINGLE_JOINT:
        temp = bone_pos_gt[:, 0]
        bone_pos_gt = temp[:, np.newaxis]
    if pose_client.model == "mpi":
        bone_pos_gt = rearrange_bones_to_mpi(bone_pos_gt)

    drone_pos_gt = np.array([simulated_value_dict['dronePos'].x_val, simulated_value_dict['dronePos'].y_val, 
                            -simulated_value_dict['dronePos'].z_val])
    drone_pos_gt = (drone_pos_gt - airsim_client.DRONE_INITIAL_POS)/100
    drone_pos_gt = drone_pos_gt[:, np.newaxis] 

    assert bone_pos_gt.shape == (3,pose_client.num_of_joints)
    assert drone_orientation_gt.shape == (3,)
    assert drone_pos_gt.shape == (3,1)

    return bone_pos_gt, drone_orientation_gt, drone_pos_gt

def get_simulator_responses(airsim_client):
    airsim_client.simPause(False) #unpause drone to take picture
    if  airsim_client.is_using_airsim:
        time.sleep(0.01)
    response = airsim_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
    airsim_client.simPause(True) #pause everything to start processing
    
    response = response[0]
    if airsim_client.is_using_airsim:
        response_poses = vector3r_arr_to_dict(response.bones)
        response_image = response.image_data_uint8
    return response_image, response_poses


def airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager):
    """
    Description: 
        Calls simulator to take picture and return GT values simulatenously. 

    Inputs: 
        airsim_client: an object of type VehicleClient or DroneFlightClient
        pose_client: an object of type PoseEstimationClient
        current_state: an object of type State
        file_manager: object of type FileManager
    Returns:
        image: photo taken at simulation step
    """

    response_image, response_poses = get_simulator_responses(airsim_client)
    if airsim_client.is_using_airsim:
        bone_pos_gt, drone_orientation_gt, drone_pos_gt = get_client_gt_values(airsim_client, pose_client, response_poses)
        file_manager.record_gt_pose(bone_pos_gt, airsim_client.linecount)
        file_manager.record_drone_info(drone_pos_gt, drone_orientation_gt, airsim_client.linecount)

        #multirotor_state = airsim_client.getMultirotorState()
        #estimated_state =  multirotor_state.kinematics_estimated
        #drone_pos_est = estimated_state.position

        current_state.store_frame_parameters(bone_pos_gt, drone_orientation_gt, drone_pos_gt)
    else:
        bone_pos_gt, drone_transformation_matrix = airsim_client.read_frame_gt_values()
        current_state.store_frame_transformation_matrix_joint_gt(bone_pos_gt, drone_transformation_matrix)

    #figure out a way to convert png bytes to float array
    image = response_image
    current_state.update_anim_time(airsim_client.getAnimationTime())
    #image_buffer =  Image.frombytes(mode="I", size=(airsim_client.SIZE_X, airsim_client.SIZE_Y), data=image)
   # print(image_buffer.shape)
    #image_buffer.show()

    return image

def take_photo(airsim_client, pose_client, current_state, file_manager, viewpoint = ""):
    """
    Description: 
        Calls simulator to take picture and return GT values simultaneously.
        Writes picture file.

    Inputs: 
        airsim_client: an object of type VehicleClient or DroneFlightClient
        pose_client: an object of type PoseEstimationClient
        current_state: an object of type State
        file_manager: object of type FileManager
        viewpoint: int. (default: ""), the viewpoint from which the drone is looking at the person
    Returns:
        photo: photo taken at simulation step
    """
    if airsim_client.is_using_airsim:
        photo = airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        loc = file_manager.update_photo_loc(linecount=airsim_client.linecount, viewpoint=viewpoint)
        airsim.write_file(os.path.normpath(loc), photo)
        if airsim_client.linecount > 3 and pose_client.quiet:
            loc_rem = file_manager.take_photo_loc + '/img_' + str(airsim_client.linecount-2) + '.png'
            if os.path.isfile(loc_rem):
                os.remove(loc_rem)
    else:
        photo = airsim_retrieve_gt(airsim_client, pose_client, current_state, file_manager)
        file_manager.update_photo_loc(linecount=airsim_client.linecount, viewpoint=viewpoint)
    return photo
