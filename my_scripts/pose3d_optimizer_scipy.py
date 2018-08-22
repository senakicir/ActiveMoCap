from helpers import * 
from project_bones import take_bone_projection
from torch.autograd import Variable
import pose3d_optimizer as pytorch_optimizer 

def mse_loss(input_1, input_2):
    return np.sum(np.square((input_1 - input_2))) / input_1.shape[0]
def find_residuals(input_1, input_2):
    return (np.square((input_1 - input_2))).ravel()

class pose3d_calibration_scipy():
    def __init__(self):
        self.pytorch_objective = 0

    def reset(self, model, data_list, weights, loss_dict):
        self.bone_connections, _, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.data_list = data_list
        self.energy_weights = weights
        self.pltpts = {}
        self.loss_dict = loss_dict
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []

        self.pytorch_objective = pytorch_optimizer.pose3d_calibration(model)


    def forward_powell(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = [3, self.NUM_OF_JOINTS], order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0
            
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            projected_2d, _ = take_bone_projection(pose_3d, R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)

        left_bone_connections, right_bone_connections, _ = split_bone_connections(self.bone_connections)
        bonelosses = np.zeros([len(left_bone_connections),])
        for i, l_bone in enumerate(left_bone_connections):
            r_bone = right_bone_connections[i]
            left_length_of_bone = (np.sum(np.square(pose_3d[:, l_bone[0]] - pose_3d[:, l_bone[1]])))
            right_length_of_bone = (np.sum(np.square(pose_3d[:, r_bone[0]] - pose_3d[:, r_bone[1]])))
            bonelosses[i] = np.square((left_length_of_bone - right_length_of_bone))
        output["sym"] = np.sum(bonelosses)/bonelosses.shape[0]

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]/len(self.loss_dict)
            self.pltpts[loss_key].append(output[loss_key])
        
        return overall_output

    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = [3, self.NUM_OF_JOINTS], order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        left_bone_connections, right_bone_connections, _ = split_bone_connections(self.bone_connections)
        bonelosses = np.zeros([len(left_bone_connections),])
        for i, l_bone in enumerate(left_bone_connections):
            r_bone = right_bone_connections[i]
            left_length_of_bone = (np.sum(np.square(pose_3d[:, l_bone[0]] - pose_3d[:, l_bone[1]])))
            right_length_of_bone = (np.sum(np.square(pose_3d[:, r_bone[0]] - pose_3d[:, r_bone[1]])))
            bonelosses[i,] = np.square((left_length_of_bone - right_length_of_bone))
        output["sym"] += np.sum(bonelosses)/(self.NUM_OF_JOINTS-1)
        residuals = bonelosses* self.energy_weights["sym"]
    
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            projected_2d, _ = take_bone_projection(pose_3d, R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)
            current_residuals = find_residuals(projected_2d, bone_2d_)* self.energy_weights["proj"]
            residuals = np.concatenate([residuals, current_residuals])

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]/len(self.loss_dict)
            self.pltpts[loss_key].append(output[loss_key])
        
        return overall_output

    def jacobian(self, x):
        self.pytorch_objective.zero_grad()
        outputs = {}
        output = {}
        for loss_key in self.loss_dict:
            outputs[loss_key] = []
            output[loss_key] = 0

        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            R_drone_torch = torch.from_numpy(R_drone_).float()
            C_drone_torch = torch.from_numpy(C_drone_).float()
            bone_2d_torch = torch.from_numpy(bone_2d_).float()
            loss = self.pytorch_objective.forward(bone_2d_torch, R_drone_torch, C_drone_torch)
            for loss_key in self.loss_dict:
                outputs[loss_key].append(loss[loss_key])

        overall_output = Variable(torch.FloatTensor([0]))
        for loss_key in self.loss_dict:
            output[loss_key] = (sum(outputs[loss_key])/len(outputs[loss_key]))
            overall_output += self.energy_weights[loss_key]*output[loss_key]
        overall_output.backward(retain_graph = True)
        gradient_torch = self.pytorch_objective.pose3d.grad
        gradient = gradient_torch.view(-1).data.numpy()
        """
        self.pytorch_objective.zero_grad()
        count = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            R_drone_torch = torch.from_numpy(R_drone_).float()
            C_drone_torch = torch.from_numpy(C_drone_).float()
            bone_2d_torch = torch.from_numpy(bone_2d_).float()
            new_residuals = self.pytorch_objective.forward(bone_2d_torch, R_drone_torch, C_drone_torch)
            if count == 0:
                residuals = new_residuals
            else:
                residuals = torch.cat((residuals, new_residuals), 0)
            count =+ 1
       
        residuals.backward(retain_graph = True)
        gradient_torch = self.pytorch_objective.pose3d.grad"""
        return gradient

class pose3d_flight_scipy():
    def __init__(self):
        self.curr_iter = 0

    def reset(self, model, data_list, lift_list, weights, loss_dict, window_size, bone_lengths):
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.data_list = data_list
        self.lift_list = lift_list
        self.energy_weights = weights
        self.pltpts = {}
        self.loss_dict = loss_dict
        self.window_size = window_size
        self.bone_lengths = bone_lengths
        self.curr_iter = 0
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
        self.pytorch_objective = pytorch_optimizer.pose3d_flight(bone_lengths, window_size, model)

    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = [self.window_size, 3, self.NUM_OF_JOINTS], order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        #projection
        queue_index = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:

            #projection
            projected_2d, _ = take_bone_projection(pose_3d[queue_index, :, :], R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)
            current_residuals = find_residuals(projected_2d, bone_2d_)
            if queue_index == 0:
                residuals = current_residuals* self.energy_weights["proj"]
            else: 
                residuals =  np.concatenate([residuals, current_residuals])


            #smoothness
            if (queue_index != self.window_size-1 and queue_index != 0):
                output["smooth"] += mse_loss(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :]) +  mse_loss(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :])
                current_residuals = (find_residuals(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :]) + find_residuals(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :]))* self.energy_weights["smooth"]
            elif (queue_index != self.window_size-1 ):
                output["smooth"] += mse_loss(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :])
                current_residuals = find_residuals(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :]) * self.energy_weights["smooth"]
            elif (queue_index != 0):
                output["smooth"] += mse_loss(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :])
                current_residuals = find_residuals(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :]) * self.energy_weights["smooth"]
            residuals =  np.concatenate([residuals, current_residuals])

            #smooth pose
            #hip_index = self.joint_names.index('spine1')
            #hip = pose_3d[queue_index, :, hip_index]
            #temp_pose3d_t = pose_3d[queue_index, :, :] - hip[:, np.newaxis]
            #if (queue_index != self.window_size-1 and queue_index != 0):
            #    p_hip = pose_3d[queue_index+1, :, hip_index]
            #    temp_pose3d_t_p_1 = pose_3d[queue_index+1, :, :]- p_hip[:, np.newaxis]
            #    m_hip = pose_3d[queue_index-1, :, hip_index]
            #    temp_pose3d_t_m_1 = pose_3d[queue_index-1, :, :]- m_hip[:, np.newaxis]
            #    output["smoothpose"] += mse_loss(temp_pose3d_t, temp_pose3d_t_p_1) +  mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)
            #elif (queue_index != self.window_size-1 ):
            #    p_hip = pose_3d[queue_index+1, :, hip_index]
            #    temp_pose3d_t_p_1 = pose_3d[queue_index+1, :, :]- p_hip[:, np.newaxis]
            #    output["smoothpose"] += mse_loss(temp_pose3d_t, temp_pose3d_t_p_1)
            #elif (queue_index != 0):
            #    m_hip = pose_3d[queue_index-1, :, hip_index]
            #    temp_pose3d_t_m_1 = pose_3d[queue_index-1, :, :]- m_hip[:, np.newaxis]
            #    output["smoothpose"] += mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)

            #lift
            pose3d_lift_directions = self.lift_list[queue_index]

            pose_est_directions = np.zeros([3, self.NUM_OF_JOINTS-1])
            for i, bone in enumerate(self.bone_connections):
                bone_vector = pose_3d[queue_index, :, bone[0]] - pose_3d[queue_index, :, bone[1]]
                pose_est_directions[:, i] = bone_vector/(np.linalg.norm(bone_vector)+EPSILON)
            #max_z = np.max(temp_pose3d_t[2,:])
            #min_z = np.min(temp_pose3d_t[2,:])
            #normalized_pose_3d = (pose3d_lift-min_z)/(max_z - min_z)
            output["lift"] += mse_loss(pose3d_lift_directions, pose_est_directions)
            current_residuals = find_residuals(pose3d_lift_directions, pose_est_directions) * self.energy_weights["lift"]
            residuals =  np.concatenate([residuals, current_residuals])

            #bone length consistency 
            bonelosses = np.zeros([self.NUM_OF_JOINTS-1,])
            for i, bone in enumerate(self.bone_connections):
                length_of_bone = (np.sum(np.square(pose_3d[queue_index, :, bone[0]] - pose_3d[queue_index, :, bone[1]])))
                bonelosses[i] = np.square((self.bone_lengths[i] - length_of_bone))
            output["bone"] += np.sum(bonelosses)/(self.NUM_OF_JOINTS-1)
            current_residuals = bonelosses* self.energy_weights["bone"]
            residuals =  np.concatenate([residuals, current_residuals])

            queue_index += 1

        if (self.curr_iter % 5000 == 0):
            print("output", output)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]/len(self.loss_dict)
            self.pltpts[loss_key].append(output[loss_key])
        
        self.curr_iter += 1
        return overall_output
        
    def jacobian(self, x):
        self.pytorch_objective.zero_grad()
        outputs = {}
        output = {}
        for loss_key in self.loss_dict:
            outputs[loss_key] = []
            output[loss_key] = 0

        queue_index = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            pose3d_lift_directions = self.lift_list[queue_index]
            R_drone_torch = torch.from_numpy(R_drone_).float()
            C_drone_torch = torch.from_numpy(C_drone_).float()
            bone_2d_torch = torch.from_numpy(bone_2d_).float()
            loss = self.pytorch_objective.forward(bone_2d_torch, R_drone_torch, C_drone_torch, pose3d_lift_directions, queue_index)
            for loss_key in self.loss_dict:
                outputs[loss_key].append(loss[loss_key])
            queue_index =+ 1

        overall_output = Variable(torch.FloatTensor([0]))
        for loss_key in self.loss_dict:
            output[loss_key] = (sum(outputs[loss_key])/len(outputs[loss_key]))
            overall_output += self.energy_weights[loss_key]*output[loss_key]
        overall_output.backward(retain_graph = True)
        gradient_torch = self.pytorch_objective.pose3d.grad
        gradient = gradient_torch.view(-1).data.numpy()
        return gradient

