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

def fun_forward(pytorch_objective, x, new_shape):
    x_scrambled = np.reshape(a = x, newshape = new_shape, order = "C")
    pytorch_objective.init_pose3d(x_scrambled)
    overall_output = pytorch_objective.forward()
    return overall_output.data.numpy()

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
        self.M = pose_client.M
        self.pytorch_objective = pytorch_optimizer.pose3d_calibration(pose_client)
        self.result_shape = pose_client.result_shape_calib

    def forward(self, x):
        overall_output = fun_forward(self.pytorch_objective, x, self.result_shape)
        self.pltpts = self.pytorch_objective.pltpts
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
        hessian = fun_hessian(self.pytorch_objective, x, self.result_shape)
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
        self.pytorch_objective = pytorch_optimizer.pose3d_flight(pose_client)
        self.lift_bone_directions = return_lift_bone_connections(self.bone_connections)
        self.M = pose_client.M
        self.result_shape = pose_client.result_shape_flight

    def forward(self, x):
        overall_output = fun_forward(self.pytorch_objective, x, self.result_shape)
        self.pltpts = self.pytorch_objective.pltpts
        return overall_output
                
    def jacobian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape)
        return gradient

    def hessian(self, x):
        hessian = fun_hessian(self.pytorch_objective, x, self.result_shape)
        return hessian

    def mini_hessian_2(self,x):
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

    def mini_hessian(self,x):
        self.pytorch_objective.zero_grad()
        self.pytorch_objective.init_pose3d(x)
        overall_output = self.pytorch_objective.forward()
        gradient_torch_l = grad(overall_output, self.pytorch_objective.pose3d, create_graph=True)
        gradient_torch = gradient_torch_l[0]
        hessian_torch = torch.zeros((self.NUM_OF_JOINTS)*3,(self.NUM_OF_JOINTS)*3)
        for i in range(0,3):
            for j in range(0, self.NUM_OF_JOINTS):
                ele = gradient_torch[0,i,j]
                temp_l = grad(ele, self.pytorch_objective.pose3d, create_graph=True)
                temp = temp_l[0]
                hessian_torch[:, j+i*self.NUM_OF_JOINTS] = temp[0, :, :].view(-1)
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

    def forward(self, x):
        overall_output = fun_forward(self.pytorch_objective, x, self.result_shape)
        self.pltpts = self.pytorch_objective.pltpts
        return overall_output
        
    def jacobian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape)
        return gradient

    def hessian(self, x):
        hessian = fun_hessian(self.pytorch_objective, x, self.result_shape)
        return hessian

    def mini_hessian_hip(self,x):
        self.pytorch_objective.zero_grad()
        self.pytorch_objective.init_pose3d(x)
        overall_output = self.pytorch_objective.forward()
        gradient_torch_l = grad(overall_output, self.pytorch_objective.pose3d, create_graph=True)
        gradient_torch = gradient_torch_l[0]
        hessian_torch = torch.zeros((self.window_size+1)*3,(self.window_size+1)*3)
        hip_index = self.joint_names.index('spine1')
        index = 0 
        for j in range(0, self.window_size+1):
            for i in range(0,3):
                ele = gradient_torch[j,i,hip_index]
                temp_l = grad(ele, self.pytorch_objective.pose3d, create_graph=True)
                temp = temp_l[0]
                #hessian_torch[:, j*3+i] = temp[:, :, hip_index].view(-1)
                hessian_torch[:,index] = temp[:, :, hip_index].view(-1)
                index += 1
        hessian = hessian_torch.data.numpy()
        return hessian

    def mini_hessian(self,x):
        self.pytorch_objective.zero_grad()
        self.pytorch_objective.init_pose3d(x)
        overall_output = self.pytorch_objective.forward()
        gradient_torch_l = grad(overall_output, self.pytorch_objective.pose3d, create_graph=True)
        gradient_torch = gradient_torch_l[0]
        hessian_torch = torch.zeros((self.NUM_OF_JOINTS)*3,(self.NUM_OF_JOINTS)*3)
        index = 0
        for i in range(0,3):
            for j in range(0, self.NUM_OF_JOINTS):
                ele = gradient_torch[0,i,j]
                temp_l = grad(ele, self.pytorch_objective.pose3d, create_graph=True)
                temp = temp_l[0]
                #hessian_torch[:, j+i*self.NUM_OF_JOINTS] = temp[0, :, :].view(-1)
                hessian_torch[:, index] = temp[0, :, :].view(-1)
                index += 1
        hessian = hessian_torch.data.numpy()
        return hessian

    def approx_hessian(self,x):
        noise = np.random.normal(loc=0, scale=0.01, size=[self.window_size+1, 3, self.NUM_OF_JOINTS])
        gradient = fun_jacobian(self.pytorch_objective, x+noise, self.result_shape)
        jacobian = gradient[np.newaxis,:]
        return np.dot(jacobian.T, jacobian)

    