from helpers import * 
from project_bones import take_bone_projection_pytorch, Projection_Client, take_bone_projection_pytorch
import pose3d_optimizer as pytorch_optimizer 
from scipy.optimize._numdiff import approx_derivative, group_columns
from torch.autograd import grad
import pdb

def find_residuals(input_1, input_2):
    return (np.square((input_1 - input_2))).ravel()

def fun_forward(pytorch_objective, x, new_shape):
    x_scrambled = np.reshape(a = x, newshape = new_shape, order = "C")
    pytorch_objective.init_pose3d(x_scrambled)
    overall_output = pytorch_objective.forward()
    return overall_output.cpu().detach().numpy()

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
    gradient_scrambled = gradient_torch.cpu().numpy()
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
        gradient_scrambled = gradient_torch.cpu().numpy()
        gradient[ind, :] = np.reshape(a = gradient_scrambled, newshape = [multip_dim, ], order = "C")
    return gradient

def fun_hessian(pytorch_objective, x, result_shape):
    hess_size = 1
    for i in result_shape:
        hess_size *= i

    #first derivative
    pytorch_objective.zero_grad()
    pytorch_objective.init_pose3d(x)
    overall_output = pytorch_objective.forward()
    gradient_torch = grad(overall_output, pytorch_objective.pose3d, create_graph=True)
    gradient_torch_flat = gradient_torch[0].view(-1)
    
    #second derivative
    hessian_torch = torch.zeros(hess_size, hess_size)
    for ind, ele in enumerate(gradient_torch_flat):
        temp = grad(ele, pytorch_objective.pose3d, create_graph=True)
        hessian_torch[:, ind] = temp[0].view(-1)

    hessian = hessian_torch.detach().numpy()
    return hessian

class pose3d_calibration_parallel_wrapper():
    def __init__(self):
        self.pytorch_objective = 0

    def reset(self, pose_client):
        _, _, self.NUM_OF_JOINTS, _ = pose_client.model_settings()
        
        data_list = pose_client.requiredEstimationData

        projection_client = Projection_Client()
        projection_client.reset(data_list, self.NUM_OF_JOINTS)
        self.pytorch_objective = pytorch_optimizer.pose3d_calibration_parallel(pose_client, projection_client)

        self.pltpts = {}
        self.result_shape = pose_client.result_shape

    def reset_future(self, pose_client, potential_state):
        _, _, self.NUM_OF_JOINTS, _ = pose_client.model_settings()
        data_list = pose_client.requiredEstimationData
        projection_client = Projection_Client()

        #future state 
        yaw = potential_state["orientation"]
        C_drone =  potential_state["position"].copy()
        potential_pitch = potential_state["pitch"]
        future_pose = torch.from_numpy(pose_client.future_pose).float()

        potential_R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, potential_pitch+pi/2, CAMERA_YAW_OFFSET, returnTensor=True)
        potential_R_drone = euler_to_rotation_matrix(0, 0, yaw, returnTensor = True)
        potential_C_drone = torch.from_numpy(C_drone[:, np.newaxis]).float()
        potential_projected_est, _ = take_bone_projection_pytorch(future_pose, potential_R_drone, potential_C_drone, potential_R_cam)

        projection_client.reset_future(data_list, self.NUM_OF_JOINTS, potential_R_cam, potential_R_drone, potential_C_drone, potential_projected_est)
        self.pytorch_objective = pytorch_optimizer.pose3d_calibration_parallel(pose_client, projection_client)

        self.pltpts = {}
        self.result_shape = pose_client.result_shape

    def forward(self, x):
        overall_output = fun_forward(self.pytorch_objective, x, self.result_shape)
        self.pltpts = self.pytorch_objective.pltpts
        return overall_output

    def jacobian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape)
        return gradient

    def hessian(self, x):
        hessian = fun_hessian(self.pytorch_objective, x[0,:,:], self.result_shape)
        return hessian

class pose3d_online_parallel_wrapper():

    def reset(self, pose_client):
        _, _, self.NUM_OF_JOINTS, _ = pose_client.model_settings()
        data_list = pose_client.requiredEstimationData
        projection_client = Projection_Client()

        projection_client.reset(data_list, self.NUM_OF_JOINTS)
        
        if pose_client.USE_TRAJECTORY_BASIS:
            self.pytorch_objective = pytorch_optimizer.pose3d_online_parallel_traj(pose_client, projection_client, future_proj=False)
        else:
            self.pytorch_objective = pytorch_optimizer.pose3d_online_parallel(pose_client, projection_client, future_proj=False)

        self.pltpts = {}
        self.result_shape = pose_client.result_shape

    def reset_future(self, pose_client, potential_state):
        _, _, self.NUM_OF_JOINTS, _ = pose_client.model_settings()
        data_list = pose_client.requiredEstimationData
        projection_client = Projection_Client()

        #future state 
        yaw = potential_state["orientation"]
        C_drone =  potential_state["position"].copy()
        potential_pitch = potential_state["pitch"]
        future_pose = torch.from_numpy(pose_client.future_pose).float()

        potential_R_cam = euler_to_rotation_matrix (CAMERA_ROLL_OFFSET, potential_pitch+pi/2, CAMERA_YAW_OFFSET, returnTensor = True)
        potential_R_drone = euler_to_rotation_matrix(0, 0, yaw, returnTensor = True)
        potential_C_drone = torch.from_numpy(C_drone[:, np.newaxis]).float()
        potential_projected_est, _ = take_bone_projection_pytorch(future_pose, potential_R_drone, potential_C_drone, potential_R_cam)

        projection_client.reset_future(data_list, self.NUM_OF_JOINTS, potential_R_cam, potential_R_drone, potential_C_drone, potential_projected_est)

        if pose_client.USE_TRAJECTORY_BASIS:
            self.pytorch_objective = pytorch_optimizer.pose3d_online_parallel_traj(pose_client, projection_client, future_proj=True)
        else:
            self.pytorch_objective = pytorch_optimizer.pose3d_online_parallel(pose_client, projection_client, future_proj=True)

        self.pltpts = {}
        self.result_shape = pose_client.result_shape

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
