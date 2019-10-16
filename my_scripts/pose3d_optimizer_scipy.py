from helpers import * 
from project_bones import Projection_Client
import pose3d_optimizer as pytorch_optimizer 
from scipy.optimize._numdiff import approx_derivative, group_columns
from Lift_Client import Lift_Client, calculate_bone_directions, calculate_bone_directions_simple
from torch.autograd import grad
import pdb

def find_residuals(input_1, input_2):
    return (np.square((input_1 - input_2))).ravel()

def fun_forward(pytorch_objective, x, new_shape):
    x_scrambled = np.reshape(a = x, newshape = new_shape, order = "C")
    pytorch_objective.init_pose3d(x_scrambled)
    overall_output = pytorch_objective.forward()
    return overall_output.detach().cpu().numpy()

def fun_jacobian(pytorch_objective, x, new_shape, optimization_mode, FUTURE_WINDOW_SIZE):
    multip_dim = np.prod(new_shape)
    x_scrambled = np.reshape(a = x, newshape = new_shape, order = "C")
    pytorch_objective.init_pose3d(x_scrambled)
    pytorch_objective.zero_grad()
    overall_output = pytorch_objective.forward()
    overall_output.backward(create_graph=False)
    gradient_torch = pytorch_objective.pose3d.grad
    if optimization_mode == "estimate_future":
        gradient_torch[FUTURE_WINDOW_SIZE:, :, :] = 0
    gradient_scrambled = gradient_torch.detach().cpu().numpy()
    gradient = np.reshape(a = gradient_scrambled, newshape =  [multip_dim, ], order = "C")
    return gradient
 
def fun_jacobian_residuals(pytorch_objective, x, new_shape):
    multip_dim = np.prod(new_shape)
        
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

def fun_hessian(pytorch_objective, x, result_shape, device):
    hess_size = np.prod(result_shape)

    #first derivative
    pytorch_objective.zero_grad()
    pytorch_objective.init_pose3d(x)
    overall_output = pytorch_objective.forward()
    gradient_torch = grad(overall_output, pytorch_objective.pose3d, create_graph=True)
    gradient_torch_flat = gradient_torch[0].view(-1)
    
    #second derivative
    hessian_torch = torch.zeros(hess_size, hess_size).to(device)
    for ind, ele in enumerate(gradient_torch_flat):
        temp = grad(ele, pytorch_objective.pose3d, create_graph=True)
        hessian_torch[:, ind] = temp[0].view(-1)
    hessian = hessian_torch.detach().cpu().numpy()
    return hessian

class pose3d_calibration_parallel_wrapper():
    def __init__(self):
        self.pytorch_objective = 0
        self.pltpts, self.pltpts_weighted  = {}, {}

    def my_init(self, pose_client):
        self.projection_client = pose_client.projection_client
        self.device = pose_client.device
        self.bone_connections, _, _, self.hip_index = pose_client.model_settings()
        self.FUTURE_WINDOW_SIZE = pose_client.FUTURE_WINDOW_SIZE
        self.result_shape = pose_client.result_shape

    def reset_current(self, pose_client):        
        data_list = pose_client.requiredEstimationData

        self.projection_client.reset(data_list)
        self.optimization_mode = "estimate_current"

        self.pytorch_objective = pytorch_optimizer.pose3d_calibration_parallel(pose_client, self.projection_client)
        self.result_shape = pose_client.result_shape
        self.device = pose_client.device

    def reset_future(self, pose_client, potential_trajectory):
        data_list = pose_client.requiredEstimationData
        projection_client = pose_client.projection_client
        future_window = pose_client.FUTURE_WINDOW_SIZE

        #future state: this is all wrong!!!
        future_poses = torch.from_numpy(pose_client.future_poses.copy()).float()

        potential_projected_est, _ = projection_client.take_single_projection(future_poses, potential_trajectory.inv_transformation_matrix)
        projection_client.reset_future(data_list, potential_trajectory.inv_transformation_matrix, potential_projected_est)
        self.pytorch_objective = pytorch_optimizer.pose3d_calibration_parallel(pose_client, projection_client)

    def forward(self, x):
        overall_output = fun_forward(self.pytorch_objective, x, self.result_shape)
        self.pltpts = self.pytorch_objective.pltpts
        self.pltpts_weighted = self.pytorch_objective.pltpts_weighted
        return overall_output

    def jacobian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape, self.optimization_mode, 0)
        return gradient

    def hessian(self, x):
        hessian = fun_hessian(self.pytorch_objective, x[0,:,:], self.result_shape, self.device)
        return hessian

class pose3d_online_parallel_wrapper():
    def __init__(self):
        self.pytorch_objective = 0
        self.pltpts, self.pltpts_weighted  = {}, {}

    def my_init(self, pose_client):
        self.projection_client = pose_client.projection_client
        self.lift_client = pose_client.lift_client
        self.device = pose_client.device
        self.bone_connections, _, _, self.hip_index = pose_client.model_settings()
        self.FUTURE_WINDOW_SIZE = pose_client.FUTURE_WINDOW_SIZE
        self.result_shape = pose_client.result_shape


    def reset_current(self, pose_client):
        data_list = pose_client.requiredEstimationData
        self.optimization_mode="estimate_past"        
        self.projection_client.reset(data_list)
        if pose_client.USE_LIFT_TERM:
            self.lift_client.reset(pose_client.lift_pose_tensor, pose_client.poses_3d_gt)

        self.pytorch_objective = pytorch_optimizer.pose3d_online_parallel(pose_client, self.projection_client, self.lift_client, optimization_mode=self.optimization_mode)

    def reset_future(self, pose_client):
        self.optimization_mode="estimate_future"
        self.pytorch_objective = pytorch_optimizer.pose3d_online_parallel(pose_client, None, None, optimization_mode=self.optimization_mode)

    def reset_hessian(self, pose_client, potential_trajectory):
        self.optimization_mode="estimate_whole"
        data_list = pose_client.requiredEstimationData
        
        future_poses = torch.from_numpy(pose_client.future_poses.copy()).float()
        self.projection_client.reset_future(data_list, future_poses, potential_trajectory)
        
        if pose_client.USE_LIFT_TERM:
            if pose_client.LIFT_METHOD == "complex":
                potential_pose3d_lift_directions = calculate_bone_directions(future_poses, np.array(return_lift_bone_connections(self.bone_connections)), batch=True) 
            if pose_client.LIFT_METHOD == "simple":
                potential_pose3d_lift_directions = calculate_bone_directions_simple(future_poses, pose_client.boneLengths, pose_client.BONE_LEN_METHOD, np.array(self.bone_connections), self.hip_index, batch=True)
            self.lift_client.reset_future(pose_client.lift_pose_tensor, potential_pose3d_lift_directions)
            
        self.pytorch_objective = pytorch_optimizer.pose3d_online_parallel(pose_client, self.projection_client, self.lift_client, optimization_mode=self.optimization_mode)

    def forward(self, x):
        overall_output = fun_forward(self.pytorch_objective, x, self.result_shape)
        self.pltpts = self.pytorch_objective.pltpts
        self.pltpts_weighted = self.pytorch_objective.pltpts_weighted
        return overall_output

    def jacobian(self,x):
        gradient = fun_jacobian(self.pytorch_objective, x, self.result_shape, self.optimization_mode, self.FUTURE_WINDOW_SIZE)
        return gradient

    def hessian(self, x):
        start1=time.time()
        hessian = fun_hessian(self.pytorch_objective, x, self.result_shape, self.device)
        end1=time.time()
        print("Time it takes to compute hessian", end1-start1, "seconds")
        return hessian