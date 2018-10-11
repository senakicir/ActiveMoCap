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

def fun_jacobian(pytorch_objective, x, new_shape):
    multip_dim = 1
    for i in new_shape:
        multip_dim *= i
    pytorch_objective.zero_grad()
    x_scrambled = np.reshape(a = x, newshape = new_shape, order = "C")
    pytorch_objective.init_pose3d(x_scrambled)
    overall_output = pytorch_objective.forward()
    overall_output.backward(create_graph=False)
    gradient_torch = pytorch_objective.pose3d.grad
    gradient_scrambled = gradient_torch.data.numpy()
    gradient = np.reshape(a = gradient_scrambled, newshape =  [multip_dim, ], order = "C")
    return gradient

def fun_jacobian_residuals(pytorch_objective, x, new_shape):
    multip_dim = 1
    for i in new_shape:
        multip_dim *= i
        
    x_scrambled = np.reshape(a = x, newshape = new_shape, order = "C")
    pytorch_objective.init_pose3d(x_scrambled)
    overall_output = pytorch_objective.forward()
    gradient = np.zeros([overall_output.shape[0], multip_dim])

    for ind, one_residual in enumerate(overall_output):
        pytorch_objective.zero_grad()
        one_residual.backward(retain_graph=True)
        gradient_torch = pytorch_objective.pose3d.grad
        gradient_scrambled = gradient_torch.data.numpy()
        gradient[ind, :] = np.reshape(a = gradient_scrambled, newshape = [multip_dim, ], order = "C")
    #print("torch grad", gradient)
    #import pdb; pdb.set_trace()
    return gradient

def fun_hessian(pytorch_objective, x, result_shape):
    hess_size = 1
    for i in result_shape:
        hess_size *= i

    pytorch_objective.zero_grad()
    pytorch_objective.init_pose3d(x)
    overall_output = pytorch_objective.forward()
    gradient_torch = grad(overall_output, pytorch_objective.pose3d, create_graph=True)

    gradient_torch_flat = gradient_torch[0].view(-1)
    hessian_torch = torch.zeros(hess_size, hess_size)
    for ind, ele in enumerate(gradient_torch_flat):
        temp = grad(ele, pytorch_objective.pose3d, create_graph=True)
        hessian_torch[:, ind] = temp[0].view(-1)

    hessian = hessian_torch.data.numpy()
    return hessian

class pose3d_calibration_scipy():
    def __init__(self):
        self.pytorch_objective = 0

    def reset(self, pose_client):
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(pose_client.model)
        self.data_list = pose_client.requiredEstimationData_calibration
        self.energy_weights = pose_client.weights_calib
        self.pltpts = {}
        self.loss_dict = pose_client.loss_dict_calib
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
        self.M = pose_client.M
        self.pytorch_objective = pytorch_optimizer.pose3d_calibration(pose_client)
        self.result_shape = pose_client.result_shape_calib

    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = self.result_shape, order = "C")
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
            
        #pose_3d_M = np.dot(pose_3d, self.M)
        pose_3d_M = pose_3d
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            projected_2d, _ = take_bone_projection(pose_3d_M, R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])

        return overall_output

    def jacobian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape)
        return gradient

    def hessian(self, x):
        hessian = fun_hessian(self.pytorch_objective, x, self.result_shape)
        return hessian

    def jacobian_residuals(self,x):
        gradient = fun_jacobian_residuals(self.pytorch_objective, x,  self.result_shape)
        return gradient

    def jacobian_3(self,x):
        gradient = approx_derivative(self.forward, x, rel_step=None, method="2-point", sparsity=None)
        #print("approx grad", gradient)
        #import pdb; pdb.set_trace()
        return gradient

    def mini_hessian(self,x):
        self.pytorch_objective.zero_grad()
        self.pytorch_objective.init_pose3d(x)
        overall_output = self.pytorch_objective.forward()
        gradient_torch_l = grad(overall_output, self.pytorch_objective.pose3d, create_graph=True)
        gradient_torch = gradient_torch_l[0]
        hessian_torch = torch.zeros(3,3)
        hip_index = self.joint_names.index('spine1')
        for i in range(0,3):
            ele = gradient_torch[i,hip_index]
            temp_l = grad(ele, self.pytorch_objective.pose3d, create_graph=True)
            temp = temp_l[0]
            hessian_torch[:, i] = temp[:, hip_index]
        hessian = hessian_torch.data.numpy()

        return hessian

    def approx_hessian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape)
        jacobian =  gradient[:, np.newaxis]
        hessian = np.dot(jacobian, jacobian.T)
        return hessian

class pose3d_flight_scipy():

    def reset(self, pose_client):
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(pose_client.model)
        self.data_list = pose_client.requiredEstimationData
        self.lift_list = pose_client.liftPoseList
        self.energy_weights = pose_client.weights_flight
        self.pltpts = {}
        self.loss_dict = pose_client.loss_dict_flight
        self.window_size = pose_client.FLIGHT_WINDOW_SIZE
        self.bone_lengths = pose_client.boneLengths
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []
        self.pytorch_objective = pytorch_optimizer.pose3d_flight(pose_client)
        self.lift_bone_directions = return_lift_bone_connections(self.bone_connections)
        self.M = pose_client.M
        self.result_shape = pose_client.result_shape_flight

    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = self.result_shape, order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        queue_index = 0 #0'th pose is t+1, 1 is t, etc. 

        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            pose_3d_curr = pose_3d[queue_index, :, :]
            if queue_index != 0: #future pose has no projection or lift term
                #pose_3d_curr = np.dot(pose_3d[queue_index, :, :], self.M)
                #projection
                projected_2d, _ = take_bone_projection(pose_3d_curr, R_drone_, C_drone_)
                output["proj"] += mse_loss(projected_2d, bone_2d_)

                #lift
                pose3d_lift_directions = self.lift_list[queue_index]
                pose_est_directions = np.zeros([3, len(self.lift_bone_directions)])
                for i, bone in enumerate(self.lift_bone_directions):
                    bone_vector = pose_3d_curr[:,bone[0]] -pose_3d_curr[:,bone[1]]
                    pose_est_directions[:, i] = bone_vector/(np.linalg.norm(bone_vector)+EPSILON)
                output["lift"] += mse_loss(pose3d_lift_directions, pose_est_directions)

            #bone length consistency 
            bonelosses = np.zeros([self.NUM_OF_JOINTS-1,])
            for i, bone in enumerate(self.bone_connections):
                length_of_bone = (np.sum(np.square(pose_3d_curr[:,bone[0]] - pose_3d_curr[:,bone[1]])))
                bonelosses[i] = np.square((self.bone_lengths[i] - length_of_bone))
            output["bone"] += np.sum(bonelosses)/(self.NUM_OF_JOINTS-1)
            
            #smoothness
            pose_vel = np.zeros([2, 3, self.NUM_OF_JOINTS])
            if (queue_index != 0 or queue_index != 1):
                #find velocities
                pose_vel[0]  = pose_3d[queue_index-1, :, :]- pose_3d[queue_index-2, :, :]
                pose_vel[1]  = pose_3d[queue_index, :, :]- pose_3d[queue_index-1, :, :]
                #want change of velocities to be constant
                output["smooth"] += mse_loss(pose_vel[0] , pose_vel[1])

            queue_index += 1

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])
        
        return overall_output
        
    def jacobian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape)
        return gradient

    def hessian(self, x):
        hessian = fun_hessian(self.pytorch_objective, x, self.result_shape)
        return hessian

    def mini_hessian(self,x):
        self.pytorch_objective.zero_grad()
        self.pytorch_objective.init_pose3d(x)
        overall_output = self.pytorch_objective.forward()
        gradient_torch_l = grad(overall_output, self.pytorch_objective.pose3d, create_graph=True)
        gradient_torch = gradient_torch_l[0]
        hessian_torch = torch.zeros((self.window_size+1)*3,(self.window_size+1)*3)
        hip_index = self.joint_names.index('spine1')
        for j in range(0, self.window_size+1):
            for i in range(0,3):
                ele = gradient_torch[j,i,hip_index]
                temp_l = grad(ele, self.pytorch_objective.pose3d, create_graph=True)
                temp = temp_l[0]
                hessian_torch[:, j*3+i] = temp[:, :, hip_index].view(-1)
        hessian = hessian_torch.data.numpy()
        return hessian

    def jacobian_residuals(self,x):
        gradient = fun_jacobian_residuals(self.pytorch_objective, x, self.result_shape)
        return gradient

    def approx_hessian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape)
        jacobian =  gradient[:, np.newaxis]
        return np.dot(jacobian, jacobian.T)

class pose3d_future():

    def reset(self, pose_client, potential_state):
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(pose_client.model)
        self.data_list = pose_client.requiredEstimationData
        self.lift_list = pose_client.liftPoseList
        self.energy_weights = pose_client.weights_future
        self.pltpts = {}
        self.loss_dict = pose_client.loss_dict_future
        self.window_size = pose_client.FLIGHT_WINDOW_SIZE
        self.bone_lengths = pose_client.boneLengths
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []

        self.lift_bone_directions = return_lift_bone_connections(self.bone_connections)
        self.result_shape = pose_client.result_shape_flight

        #future state 
        yaw = potential_state["orientation"]
        self.potential_R_drone = euler_to_rotation_matrix(0, 0, yaw)
        C_drone =  potential_state["position"]
        self.potential_C_drone = C_drone[:, np.newaxis]
        self.potential_projected_est, _ = take_bone_projection(pose_client.future_pose, self.potential_R_drone, self.potential_C_drone)

        #torch for jacobian
        self.pytorch_objective = pytorch_optimizer.pose3d_future(pose_client, self.potential_R_drone, self.potential_C_drone)


    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = self.result_shape, order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        queue_index = 0 #0'th pose is t+1, 1 is t, etc. 
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            pose_3d_curr = pose_3d[queue_index, :, :]

            if queue_index != 0: #future pose has no lift term but a special projection
                #projection
                projected_2d, _ = take_bone_projection(pose_3d_curr, R_drone_, C_drone_)
                output["proj"] += mse_loss(projected_2d, bone_2d_)

                current_res = find_residuals(projected_2d, self.potential_projected_est)* self.energy_weights["proj"]
                residuals = np.concatenate([residuals, current_res])

                #lift
                pose3d_lift_directions = self.lift_list[queue_index]
                pose_est_directions = np.zeros([3, len(self.lift_bone_directions)])
                for i, bone in enumerate(self.lift_bone_directions):
                    bone_vector = pose_3d_curr[:,bone[0]] -pose_3d_curr[:,bone[1]]
                    pose_est_directions[:, i] = bone_vector/(np.linalg.norm(bone_vector)+EPSILON)
                output["lift"] += mse_loss(pose3d_lift_directions, pose_est_directions)

                current_res = find_residuals(pose3d_lift_directions, pose_est_directions)* self.energy_weights["lift"]
                residuals = np.concatenate([residuals, current_res])

            else:
                projected_2d, _ = take_bone_projection(pose_3d_curr, self.potential_R_drone, self.potential_C_drone)
                output["proj"] += mse_loss(projected_2d, self.potential_projected_est)
                residuals = find_residuals(projected_2d, self.potential_projected_est)* self.energy_weights["proj"]

            #bone length consistency 
            bonelosses = np.zeros([self.NUM_OF_JOINTS-1,])
            for i, bone in enumerate(self.bone_connections):
                length_of_bone = (np.sum(np.square(pose_3d_curr[:,bone[0]] - pose_3d_curr[:,bone[1]])))
                bonelosses[i] = np.square((self.bone_lengths[i] - length_of_bone))
            output["bone"] += np.sum(bonelosses)/(self.NUM_OF_JOINTS-1)

            current_res = bonelosses * self.energy_weights["bone"]
            residuals = np.concatenate([residuals, current_res])

            #smoothness
            pose_vel = np.zeros([2, 3, self.NUM_OF_JOINTS])
            if (queue_index != 0 or queue_index != 1):
                #find velocities
                pose_vel[0]  = pose_3d[queue_index-1, :, :]- pose_3d[queue_index-2, :, :]
                pose_vel[1]  = pose_3d[queue_index, :, :]- pose_3d[queue_index-1, :, :]
                #want change of velocities to be constant
                output["smooth"] += mse_loss(pose_vel[0] , pose_vel[1])
    
                current_res = find_residuals(pose_vel[0], pose_vel[1])* self.energy_weights["smooth"]
                residuals = np.concatenate([residuals, current_res])

            queue_index += 1

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
            self.pltpts[loss_key].append(output[loss_key])
        
        return overall_output
        
    def jacobian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape)
        return gradient

    def hessian(self, x):
        hessian = fun_hessian(self.pytorch_objective, x, self.result_shape)
        return hessian


    def mini_hessian(self,x):
        self.pytorch_objective.zero_grad()
        self.pytorch_objective.init_pose3d(x)
        overall_output = self.pytorch_objective.forward()
        gradient_torch_l = grad(overall_output, self.pytorch_objective.pose3d, create_graph=True)
        gradient_torch = gradient_torch_l[0]
        hessian_torch = torch.zeros((self.window_size+1)*3,(self.window_size+1)*3)
        hip_index = self.joint_names.index('spine1')
        for j in range(0, self.window_size+1):
            for i in range(0,3):
                ele = gradient_torch[j,i,hip_index]
                temp_l = grad(ele, self.pytorch_objective.pose3d, create_graph=True)
                temp = temp_l[0]
                hessian_torch[:, j*3+i] = temp[:, :, hip_index].view(-1)
        hessian = hessian_torch.data.numpy()
        return hessian

    def approx_hessian(self,x):
        noise = np.random.normal(loc=0, scale=0.01, size=[self.window_size+1, 3, self.NUM])
        gradient = fun_jacobian(self.pytorch_objective+noise, x, self.result_shape)
        jacobian = gradient[np.newaxis,:]
        return np.dot(jacobian.T, jacobian)

    