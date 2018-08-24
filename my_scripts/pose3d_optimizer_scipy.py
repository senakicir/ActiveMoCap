from helpers import * 
from project_bones import take_bone_projection
from torch.autograd import Variable
import pose3d_optimizer as pytorch_optimizer 
from scipy.optimize._numdiff import approx_derivative, group_columns


def mse_loss(input_1, input_2):
    N = input_1.shape[0]*input_1.shape[1]
    return np.sum(np.square((input_1 - input_2))) / N
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

        self.pytorch_objective = pytorch_optimizer.pose3d_calibration_pytorch(model, loss_dict, weights, data_list)

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
        #bonelosses = []
        for i, l_bone in enumerate(left_bone_connections):
            r_bone = right_bone_connections[i]
            left_length_of_bone = (np.sum(np.square(pose_3d[:, l_bone[0]] - pose_3d[:, l_bone[1]])))
            right_length_of_bone = (np.sum(np.square(pose_3d[:, r_bone[0]] - pose_3d[:, r_bone[1]])))
            bonelosses[i,] = np.square((left_length_of_bone - right_length_of_bone))
            #bonelosses.append(np.square((left_length_of_bone - right_length_of_bone)))
        output["sym"] += np.sum(bonelosses)/(self.NUM_OF_JOINTS-1)
        #output["sym"] += sum(bonelosses)/(self.NUM_OF_JOINTS-1)
        residuals = bonelosses* self.energy_weights["sym"]
        #residuals = np.array(bonelosses)*self.energy_weights["sym"]

    
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            projected_2d, _ = take_bone_projection(pose_3d, R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)
            current_residuals = find_residuals(projected_2d, bone_2d_)* self.energy_weights["proj"]
            residuals = np.concatenate([residuals, current_residuals])

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]/len(self.loss_dict)
            self.pltpts[loss_key].append(output[loss_key])

        #print("scipy forward", overall_output)
        return overall_output

    def jacobian(self,x):
        self.pytorch_objective.zero_grad()
        x_scrambled = np.reshape(a = x, newshape = [3, self.NUM_OF_JOINTS], order = "C")
        self.pytorch_objective.init_pose3d(x_scrambled)
        overall_output = self.pytorch_objective.forward()
        overall_output.backward(retain_graph=True)
        gradient_torch = self.pytorch_objective.pose3d.grad
        gradient_scrambled = gradient_torch.data.numpy()
        gradient = np.reshape(a = gradient_scrambled, newshape = [3*self.NUM_OF_JOINTS,], order = "C")
        #print("torch grad", gradient)
        #import pdb; pdb.set_trace()
        return gradient

    def jacobian_2(self,x):
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
        gradient_torch = self.pytorch_objective.pose3d.grad
        gradient_scrambled = gradient_torch.data.numpy()
        gradient = np.reshape(a = gradient_scrambled, newshape = [3*self.NUM_OF_JOINTS,], order = "C")
        return gradient

    def jacobian_3(self,x):
        gradient = approx_derivative(self.forward, x, rel_step=None, method="2-point", sparsity=None)
        #print("approx grad", gradient)
        #import pdb; pdb.set_trace()
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
        self.pytorch_objective = pytorch_optimizer.pose3d_flight_pytorch(model, bone_lengths, window_size, loss_dict, weights, data_list, lift_list)

    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = [self.window_size, 3, self.NUM_OF_JOINTS], order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        queue_index = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            #projection
            projected_2d, _ = take_bone_projection(pose_3d[queue_index, :, :], R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)
            current_residuals = find_residuals(projected_2d, bone_2d_)* self.energy_weights["proj"]
            if queue_index == 0:
                residuals = current_residuals
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

            #bone length consistency 
            bonelosses = np.zeros([self.NUM_OF_JOINTS-1,])
            for i, bone in enumerate(self.bone_connections):
                length_of_bone = (np.sum(np.square(pose_3d[queue_index, :, bone[0]] - pose_3d[queue_index, :, bone[1]])))
                bonelosses[i] = np.square((self.bone_lengths[i] - length_of_bone))
            output["bone"] += np.sum(bonelosses)/(self.NUM_OF_JOINTS-1)
            current_residuals = bonelosses* self.energy_weights["bone"]
            residuals =  np.concatenate([residuals, current_residuals])

            #lift
            pose3d_lift_directions = self.lift_list[queue_index]
            pose_est_directions = np.zeros([3, self.NUM_OF_JOINTS-1])
            for i, bone in enumerate(self.bone_connections):
                bone_vector = pose_3d[queue_index, :, bone[0]] - pose_3d[queue_index, :, bone[1]]
                pose_est_directions[:, i] = bone_vector/(np.linalg.norm(bone_vector)+EPSILON)
            output["lift"] += mse_loss(pose3d_lift_directions, pose_est_directions)
            current_residuals = find_residuals(pose3d_lift_directions, pose_est_directions) * self.energy_weights["lift"]
            residuals =  np.concatenate([residuals, current_residuals])

            queue_index += 1

        #if (self.curr_iter % 5000 == 0):
         #   print("output", output)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]/len(self.loss_dict)
            self.pltpts[loss_key].append(output[loss_key])
        
        self.curr_iter += 1
        print("scipy forward", output, overall_output)
        return overall_output
        
    def jacobian(self,x):
        self.pytorch_objective.zero_grad()
        x_scrambled = np.reshape(a = x, newshape = [self.window_size, 3, self.NUM_OF_JOINTS], order = "C")
        self.pytorch_objective.init_pose3d(x_scrambled)
        overall_output = self.pytorch_objective.forward()
        overall_output.backward(retain_graph=True)
        gradient_torch = self.pytorch_objective.pose3d.grad
        gradient_scrambled = gradient_torch.data.numpy()
        gradient = np.reshape(a = gradient_scrambled, newshape = [self.window_size*3*self.NUM_OF_JOINTS,], order = "C")
        #print("torch grad", gradient)
        #import pdb; pdb.set_trace()
        return gradient

    def jacobian_2(self,x):
        self.pytorch_objective.zero_grad()
        queue_index = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            pose3d_lift_directions = torch.from_numpy(self.lift_list[queue_index]).float()
            R_drone_torch = torch.from_numpy(R_drone_).float()
            C_drone_torch = torch.from_numpy(C_drone_).float()
            bone_2d_torch = torch.from_numpy(bone_2d_).float()

            new_residuals = self.pytorch_objective.forward(bone_2d_torch, R_drone_torch, C_drone_torch, pose3d_lift_directions, queue_index)
            if queue_index == 0:
                residuals = new_residuals
            else:
                residuals = torch.cat((residuals, new_residuals), 0)
            queue_index =+ 1
        residuals.backward(retain_graph = True)
        gradient_torch = self.pytorch_objective.pose3d.grad
        gradient_scrambled = gradient_torch.data.numpy()
        gradient = np.reshape(a = gradient_scrambled, newshape = [self.window_size*3*self.NUM_OF_JOINTS,], order = "C")
        return gradient

    def jacobian_3(self,x):
        gradient = approx_derivative(self.forward, x, rel_step=None, method="2-point", sparsity=None)
        #print("approx grad", gradient)
        #import pdb; pdb.set_trace()
        return gradient