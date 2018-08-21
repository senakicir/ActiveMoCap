from helpers import *
from State import *
from NonAirSimClient import *
from pose3d_optimizer import *
from project_bones import *
from determine_positions import *
from SillyNet import *


def pose_test(parameters, energy_parameters):

    #set up folder names, plot names, file names etc.
    ANIMATION_NUM = parameters["ANIMATION_NUM"]
    test_set_name = parameters["TEST_SET_NAME"]
    folder_names = parameters["FOLDER_NAMES"]
    foldernames_anim = folder_names[ANIMATION_NUM]
    test_set_name = parameters["TEST_SET_NAME"]
    plot_loc = foldernames_anim["superimposed_images"]
    filename_bones = 'test_sets/'+test_set_name+'/groundtruth.txt'
    filename_others = 'test_sets/'+test_set_name+'/a_flight.txt'

    client = NonAirSimClient(filename_bones, filename_others)
    #net = SillyNet().cuda()
    #net.train(False)
    #net.load_state_dict(torch.load('SillyNetWeights.pth'))

    #init energy parameters
    client.lr = energy_parameters["LR_MU"][0]
    client.mu = energy_parameters["LR_MU"][1]
    client.iter = energy_parameters["ITER"]
    client.weights = energy_parameters["WEIGHTS"]
    client.model = parameters["MODEL"]
    #THESE ARE WRONG
    if client.model =="mpi":
        client.boneLengths = torch.FloatTensor([[ 0.0187],[ 0.4912],[ 0.3634],[ 0.0360],[ 0.0175],[ 0.4542],[ 0.3922],[ 0.0338],[ 0.3812],[ 0.0000],[ 0.1317],[ 0.0000],[ 0.1017],[ 0.0620],[ 0.1946],[ 0.0015],[ 0.1054],[ 0.0521],[ 0.1134],[ 0.0014]])
    else:
        client.boneLengths = torch.FloatTensor()
    #choose WINDOW_SIZE number of frames starting from 20
    unreal_positions_list, bone_pos_3d_GT_list = client.choose10Frames(20)
    
    #save the R_drone, C_drone, projected joints and 3d positions together for each frame
    for ind, unreal_positions in enumerate(unreal_positions_list):
        bone_pos_3d_GT = bone_pos_3d_GT_list[ind]
        bone_connections, joint_names, num_of_joints, bone_pos_3d_GT = model_settings(client.model, bone_pos_3d_GT)
        bone_2d, _ = determine_2d_positions(0, True, unreal_positions, bone_pos_3d_GT)

        R_drone = Variable(euler_to_rotation_matrix(unreal_positions[DRONE_ORIENTATION_IND, 0], unreal_positions[DRONE_ORIENTATION_IND, 1], unreal_positions[DRONE_ORIENTATION_IND, 2], returnTensor=True), requires_grad = False)
        C_drone = Variable(torch.FloatTensor([[unreal_positions[DRONE_POS_IND, 0]],[unreal_positions[DRONE_POS_IND, 1]],[unreal_positions[DRONE_POS_IND, 2]]]), requires_grad = False)

        #bone_2d = (bone_2d - torch.mean(bone_2d, dim=1).unsqueeze(1))/torch.std(bone_2d, dim=1).unsqueeze(1)
        #hip_2d = bone_2d[:, 0].unsqueeze(1)
        #bone_2d = bone_2d - hip_2d

        #silly_output_temp = net(bone_2d.view(-1, 2*num_of_joints).cuda())
        #silly_output_temp = silly_output_temp.view(1, 3,num_of_joints)
        #silly_output_temp = (silly_output_temp - torch.mean(silly_output_temp, dim=2).unsqueeze(2))/torch.std(silly_output_temp, dim=2).unsqueeze(2)
        #hip_new = silly_output_temp[:, :, 0].unsqueeze(2)
        #pose3d_silly = silly_output_temp - hip_new
        pose3d_silly = 0

        #for any frame that is not the last one, save gt 3d position + some noise
        #the last frame is initialized with backprojection
        if ind != client.WINDOW_SIZE-1:
            pose3d_ = torch.from_numpy(bone_pos_3d_GT)
            m = torch.distributions.normal.Normal(torch.zeros(3,num_of_joints), 0.15*torch.ones(3,num_of_joints))
            noise = m.sample().double() 
            pose3d_ =  pose3d_ + noise
            prev_pose = pose3d_
        else: 
            pose3d_ = prev_pose
            #pose3d_ = take_bone_backprojection_pytorch(bone_2d, R_drone, C_drone)

        client.addNewFrame(bone_2d, R_drone, C_drone, pose3d_, pose3d_silly)

    objective = pose3d_flight(client.boneLengths, client.WINDOW_SIZE, client.model)
    optimizer = torch.optim.SGD(objective.parameters(), lr = client.lr, momentum=client.mu)
    
    #plot "before" pictures
    for frame_ind in range(0, client.WINDOW_SIZE):
        error_3d = np.mean(np.linalg.norm(bone_pos_3d_GT_list[frame_ind] - client.poseList_3d[client.WINDOW_SIZE-frame_ind-1].data.numpy(), axis=0))
        plot_drone_and_human(bone_pos_3d_GT_list[frame_ind],  client.poseList_3d[client.WINDOW_SIZE-frame_ind-1].data.numpy(), plot_loc, frame_ind, error_3d, custom_name="before_")
        #plot_drone_and_human(client.sillyPoseList[frame_ind].data.squeeze().cpu().numpy(),  client.sillyPoseList[frame_ind].data.squeeze().cpu().numpy(), plot_loc, frame_ind, error_3d, custom_name="silly_res_")


    pltpts = {}
    for loss_key in LOSSES:
        pltpts[loss_key] = np.zeros([client.iter])

    #init all 3d poses
    for queue_index, pose3d_ in enumerate(client.poseList_3d):
        objective.init_pose3d(pose3d_, queue_index)

    for i in range(client.iter):
        if i%100 == 0:
            print(i)

        def closure():
            outputs = {}
            output = {}
            for loss_key in LOSSES:
                outputs[loss_key] = []
                output[loss_key] = 0
                
            optimizer.zero_grad()
            objective.zero_grad()

            #find losses for all energy functions here
            queue_index = 0
            for bone_2d_, R_drone_, C_drone_ in client.requiredEstimationData:
                #pose3d_silly = client.sillyPoseList[queue_index]
                pose3d_silly = 0
                loss = objective.forward(bone_2d_, R_drone_, C_drone_, pose3d_silly, queue_index)
                for loss_key in LOSSES:
                    outputs[loss_key].append(loss[loss_key])
                queue_index += 1

            #weighted average of all energy functions is taken here    
            overall_output = Variable(torch.FloatTensor([0]))
            for loss_key in LOSSES:
                output[loss_key] = (sum(outputs[loss_key])/len(outputs[loss_key]))
                overall_output += client.weights[loss_key]*output[loss_key]/len(LOSSES)
                pltpts[loss_key][i] = output[loss_key].data.numpy() 
               
            overall_output.backward(retain_graph = True)
            return overall_output
        optimizer.step(closure)

    P_world = objective.pose3d.data.numpy()

    plot_optimization_losses(pltpts, plot_loc, 0, False)

    #plot "after" pictures
    for frame_ind in range(0, client.WINDOW_SIZE):
        error_3d = np.mean(np.linalg.norm(bone_pos_3d_GT_list[frame_ind] - P_world[frame_ind,:,:], axis=0))
        plot_drone_and_human(bone_pos_3d_GT_list[frame_ind],  P_world[frame_ind,:,:], plot_loc, frame_ind, error_3d, "after_")

if __name__ == "__main__":
    #animations = [0,1,2,3]
    animations = [2]

    test_set = {}
    for animation_num in animations:
        test_set[animation_num] = TEST_SETS[animation_num]

    file_names, folder_names, f_notes_name = reset_all_folders(animations)
    parameters = {"FILE_NAMES": file_names, "FOLDER_NAMES": folder_names, "MODEL": "mpi"}
    weights_ = {'proj': 0.08, 'smooth': 0.5, 'bone': 0.3, 'smoothpose': 0.01}#, 'silly': 0.1}
    weights = {}
    weights_sum = sum(weights_.values())
    for loss_key in LOSSES:
        weights[loss_key] = weights_[loss_key]/weights_sum

    energy_parameters = {"LR_MU": [4, 0.8], "ITER": 5000, "WEIGHTS": weights}
    fill_notes(f_notes_name, parameters, energy_parameters)    #create this function

    for animation_num, test_set in test_set.items():
        start = time.time()
        parameters["ANIMATION_NUM"]= animation_num
        parameters["TEST_SET_NAME"]= test_set
        pose_test(parameters, energy_parameters)
        print("One minimization took ", time.time()-start, " seconds")