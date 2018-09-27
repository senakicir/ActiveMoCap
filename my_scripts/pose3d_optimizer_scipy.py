from helpers import * 
from project_bones import take_bone_projection
from torch.autograd import Variable, grad
import pose3d_optimizer as pytorch_optimizer 
from scipy.optimize._numdiff import approx_derivative, group_columns


def mse_loss(input_1, input_2):
    N = np.prod(input_1.shape)
    return np.sum(np.square((input_1 - input_2))) / N
def find_residuals(input_1, input_2):
    return (np.square((input_1 - input_2))).ravel()
def cauchy_loss(input_1, input_2):
    N = np.prod(input_1.shape)
    b = 1000
    sigma = input_1 - input_2
    C = (b*b)*np.log(1+np.square(sigma)/(b*b))
    return np.sum(C)/N

class pose3d_calibration_scipy():
    def __init__(self):
        self.pytorch_objective = 0

    def reset(self, model, data_list, weights, loss_dict, M):
        self.bone_connections, _, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.data_list = data_list
        self.energy_weights = weights
        self.pltpts = {}
        self.loss_dict = loss_dict
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
        self.M = M
        self.pytorch_objective = pytorch_optimizer.pose3d_calibration_pytorch(model, loss_dict, weights, data_list, M)
        self.pytorch_objective_toy = pytorch_optimizer.toy_example(model, loss_dict, weights, data_list, M)

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
            overall_output += self.energy_weights[loss_key]*output[loss_key]
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
        #residuals = bonelosses* self.energy_weights["sym"]
    
        #pose_3d_M = np.dot(pose_3d, self.M)
        pose_3d_M = pose_3d
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            projected_2d, _ = take_bone_projection(pose_3d_M, R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)
            #current_residuals = find_residuals(projected_2d, bone_2d_)* self.energy_weights["proj"]
            #residuals = np.concatenate([residuals, current_residuals])

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])

        return overall_output

    def jacobian(self,x):
        self.pytorch_objective.zero_grad()
        x_scrambled = np.reshape(a = x, newshape = [3, self.NUM_OF_JOINTS], order = "C")
        self.pytorch_objective.init_pose3d(x_scrambled)
        overall_output = self.pytorch_objective.forward()
        overall_output.backward(create_graph=False)
        gradient_torch = self.pytorch_objective.pose3d.grad
        gradient_scrambled = gradient_torch.data.numpy()
        #gradient_torch = grad(overall_output, self.pytorch_objective.pose3d, create_graph=False)
        #gradient_scrambled = gradient_torch[0].data.numpy()
        gradient = np.reshape(a = gradient_scrambled, newshape = [3*self.NUM_OF_JOINTS,], order = "C")
        return gradient

    def hessian(self, x):
        self.pytorch_objective.zero_grad()
        self.pytorch_objective.init_pose3d(x)
        overall_output = self.pytorch_objective.forward()
        gradient_torch = grad(overall_output, self.pytorch_objective.pose3d, create_graph=True)
        gradient_torch_flat = gradient_torch[0].view(-1)
        hessian_torch = torch.zeros(3*self.NUM_OF_JOINTS, 3*self.NUM_OF_JOINTS)
        for ind, ele in enumerate(gradient_torch_flat):
            temp = grad(ele, self.pytorch_objective.pose3d, create_graph=True)
            hessian_torch[:, ind] = temp[0].view(-1)

        #overall_output.backward(create_graph=True, retain_graph=True)
        #gradient_torch = self.pytorch_objective.pose3d.grad
        #gradient_torch.backward(gradient=torch.ones(gradient_torch.size()), create_graph=False, retain_graph=False)
        #hessian_torch = self.pytorch_objective.pose3d.grad
        hessian = hessian_torch.data.numpy()

        return hessian

    def hessian_toy(self, x):
        self.pytorch_objective_toy.zero_grad()
        self.pytorch_objective_toy.init_pose3d(x)
        overall_output = self.pytorch_objective_toy.forward()
        gradient_torch = grad(overall_output, self.pytorch_objective_toy.pose3d, create_graph=True)
        gradient_torch_flat = gradient_torch[0].view(-1)
        jacobian = gradient_torch_flat
        hessian_torch = torch.zeros(4,4)
        for ind, ele in enumerate(gradient_torch_flat):
            temp = grad(ele, self.pytorch_objective_toy.pose3d, create_graph=True)
            hessian_torch[:, ind] = temp[0].view(-1)
        hessian = hessian_torch.data.numpy()
        return jacobian, hessian

    def jacobian_residuals(self,x):
        x_scrambled = np.reshape(a = x, newshape = [3, self.NUM_OF_JOINTS], order = "C")
        self.pytorch_objective.init_pose3d(x_scrambled)
        overall_output = self.pytorch_objective.forward()
        gradient = np.zeros([overall_output.shape[0], 3*self.NUM_OF_JOINTS])
        for ind, one_residual in enumerate(overall_output):
            self.pytorch_objective.zero_grad()
            one_residual.backward(retain_graph=True)
            gradient_torch = self.pytorch_objective.pose3d.grad
            gradient_scrambled = gradient_torch.data.numpy()
            gradient[ind, :] = np.reshape(a = gradient_scrambled, newshape = [3*self.NUM_OF_JOINTS,], order = "C")
        #print("torch grad", gradient)
        #import pdb; pdb.set_trace()
        return gradient

    def jacobian_3(self,x):
        gradient = approx_derivative(self.forward, x, rel_step=None, method="2-point", sparsity=None)
        #print("approx grad", gradient)
        #import pdb; pdb.set_trace()
        return gradient

class pose3d_flight_scipy():
    def __init__(self):
        self.curr_iter = 0

    def reset(self, model, data_list, lift_list, weights, loss_dict, window_size, bone_lengths, M):
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.data_list = data_list
        self.lift_list = lift_list
        self.energy_weights = weights
        self.pltpts = {}
        self.loss_dict = loss_dict
        self.window_size = window_size
        self.bone_lengths = bone_lengths
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
        self.pytorch_objective = pytorch_optimizer.pose3d_flight_pytorch(model, bone_lengths, window_size, loss_dict, weights, data_list, lift_list, M)
        self.lift_bone_directions = return_lift_bone_connections(self.bone_connections)
        self.M = M

    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = [self.window_size, 3, self.NUM_OF_JOINTS], order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        queue_index = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            #pose_3d_M = np.dot(pose_3d[queue_index, :, :], self.M)
            pose_3d_M = pose_3d[queue_index, :, :]
            #projection
            projected_2d, _ = take_bone_projection(pose_3d_M, R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)
            #current_residuals = find_residuals(projected_2d, bone_2d_)* self.energy_weights["proj"]
            #if queue_index == 0:
            #    residuals = current_residuals
            #else: 
            #    residuals =  np.concatenate([residuals, current_residuals])

            #smoothness
            #if (queue_index != self.window_size-1 and queue_index != 0):
            #    output["smooth"] += mse_loss(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :]) +  mse_loss(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :])
            #    current_residuals = (find_residuals(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :]) + find_residuals(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :]))* self.energy_weights["smooth"]
            #elif (queue_index != self.window_size-1 ):
            #    output["smooth"] += mse_loss(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :])
             #   current_residuals = find_residuals(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :]) * self.energy_weights["smooth"]
            #elif (queue_index != 0):
            #    output["smooth"] += mse_loss(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :])
            #    current_residuals = find_residuals(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :]) * self.energy_weights["smooth"]
            #residuals =  np.concatenate([residuals, current_residuals])

            #smoothness
            #find velocities
            pose_vel = np.zeros([2, 3, self.NUM_OF_JOINTS])
            if (queue_index != 0 or queue_index != 1):
                pose_vel[0]  = pose_3d[queue_index-1, :, :]- pose_3d[queue_index-2, :, :]
                pose_vel[1]  = pose_3d[queue_index, :, :]- pose_3d[queue_index-1, :, :]
                output["smooth"] += mse_loss(pose_vel[0] , pose_vel[1])

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
                length_of_bone = (np.sum(np.square(pose_3d_M[:,bone[0]] - pose_3d_M[:,bone[1]])))
                bonelosses[i] = np.square((self.bone_lengths[i] - length_of_bone))
            output["bone"] += np.sum(bonelosses)/(self.NUM_OF_JOINTS-1)
            #current_residuals = bonelosses* self.energy_weights["bone"]
            #residuals =  np.concatenate([residuals, current_residuals])

            #lift
            pose3d_lift_directions = self.lift_list[queue_index]
            pose_est_directions = np.zeros([3, len(self.lift_bone_directions)])
            for i, bone in enumerate(self.lift_bone_directions):
                bone_vector = pose_3d_M[:,bone[0]] -pose_3d_M[:,bone[1]]
                pose_est_directions[:, i] = bone_vector/(np.linalg.norm(bone_vector)+EPSILON)
            output["lift"] += mse_loss(pose3d_lift_directions, pose_est_directions)
            #current_residuals = find_residuals(pose3d_lift_directions, pose_est_directions) * self.energy_weights["lift"]
            #residuals =  np.concatenate([residuals, current_residuals])
            queue_index += 1

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])
        
        return overall_output
        
    def jacobian(self,x):
        self.pytorch_objective.zero_grad()
        x_scrambled = np.reshape(a = x, newshape = [self.window_size, 3, self.NUM_OF_JOINTS], order = "C")
        self.pytorch_objective.init_pose3d(x_scrambled)
        overall_output = self.pytorch_objective.forward()
        overall_output.backward(create_graph=False)
        gradient_torch = self.pytorch_objective.pose3d.grad
        gradient_scrambled = gradient_torch.data.numpy()
       # gradient_torch = grad(overall_output, self.pytorch_objective.pose3d, create_graph=False)
        #gradient_scrambled = gradient_torch[0].data.numpy()
        gradient = np.reshape(a = gradient_scrambled, newshape = [self.window_size*3*self.NUM_OF_JOINTS,], order = "C")
        return gradient

    def hessian(self, x):
        self.pytorch_objective.zero_grad()
        self.pytorch_objective.init_pose3d(x)
        overall_output = self.pytorch_objective.forward()
        gradient_torch = grad(overall_output, self.pytorch_objective.pose3d, create_graph=True)
        gradient_torch_flat = gradient_torch[0].view(-1)
        hessian_torch = torch.zeros(self.window_size*3*self.NUM_OF_JOINTS, self.window_size*3*self.NUM_OF_JOINTS)
        for ind, ele in enumerate(gradient_torch_flat):
            temp = grad(ele, self.pytorch_objective.pose3d, create_graph=True)
            hessian_torch[:, ind] = temp[0].view(-1)

        #overall_output.backward(create_graph=True, retain_graph=True)
        #gradient_torch = self.pytorch_objective.pose3d.grad
        #gradient_torch.backward(gradient=torch.ones(gradient_torch.size()), create_graph=False, retain_graph=False)
        #hessian_torch = self.pytorch_objective.pose3d.grad
        hessian = hessian_torch.data.numpy()

        return hessian

    def jacobian_residuals(self,x):
        x_scrambled = np.reshape(a = x, newshape = [self.window_size, 3, self.NUM_OF_JOINTS], order = "C")
        self.pytorch_objective.init_pose3d(x_scrambled)
        overall_output = self.pytorch_objective.forward()
        gradient = np.zeros([overall_output.shape[0], self.window_size*3*self.NUM_OF_JOINTS])
        for ind, one_residual in enumerate(overall_output):
            self.pytorch_objective.zero_grad()
            one_residual.backward(retain_graph=True)
            gradient_torch = self.pytorch_objective.pose3d.grad
            gradient_scrambled = gradient_torch.data.numpy()
            gradient[ind, :] = np.reshape(a = gradient_scrambled, newshape = [self.window_size*3*self.NUM_OF_JOINTS,], order = "C")
        #print("torch grad", gradient)
        #import pdb; pdb.set_trace()
        return gradient