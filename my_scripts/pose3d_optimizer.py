from helpers import * 
from project_bones import take_bone_projection_pytorch
from torch.autograd import Variable

def mse_loss(input_1, input_2):
    N = input_1.data.nelement()
    return torch.sum(torch.pow((input_1 - input_2),2)) / N

def blake_zisserman_loss(input_1, input_2):
    N = input_1.data.nelement()
    C = -torch.log(torch.exp(-torch.pow((input_1-input_2),2))+EPSILON)
    return torch.sum(C)/N

def cauchy_loss(input_1, input_2):
    N = input_1.data.nelement()
    b = 1000
    sigma = input_1 - input_2
    C = (b*b)*torch.log(1+torch.pow(sigma,2)/(b*b))
    return torch.sum(C)/N

def find_residuals(input_1, input_2):
    return (torch.pow((input_1 - input_2),2)).view(-1)

class pose3d_calibration_2(torch.nn.Module):
    def __init__(self, model):
        super(pose3d_calibration_2, self).__init__()
        self.bone_connections, _, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.pose3d = torch.nn.Parameter(torch.zeros([3, self.NUM_OF_JOINTS]), requires_grad=True)

    def forward(self, pose_2d, R_drone, C_drone):
        outputs = {}
        for loss in CALIBRATION_LOSSES:
            outputs[loss] = 0

        left_bone_connections, right_bone_connections, _ = split_bone_connections(self.bone_connections)
        bonelosses = Variable(torch.zeros([len(left_bone_connections),1]), requires_grad = False)
        for i, l_bone in enumerate(left_bone_connections):
            r_bone = right_bone_connections[i]
            left_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, l_bone[0]] - self.pose3d[:, l_bone[1]], 2)))
            right_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, r_bone[0]] - self.pose3d[:, r_bone[1]], 2)))
            bonelosses[i] = torch.pow((left_length_of_bone - right_length_of_bone),2)
        outputs["sym"] = torch.sum(bonelosses)/bonelosses.data.nelement()

        projected_2d, _ = take_bone_projection_pytorch(self.pose3d, R_drone, C_drone)

        outputs["proj"] = mse_loss(projected_2d, pose_2d)
        
        outputs["proj"] += mse_loss

        return outputs
    
    def init_pose3d(self, pose3d_):
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_flight_2(torch.nn.Module):

    def __init__(self, bone_lengths_, window_size_, model):
        super(pose3d_flight_2, self).__init__()
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.window_size = window_size_
        self.pose3d = torch.nn.Parameter(torch.zeros([self.window_size, 3, self.NUM_OF_JOINTS]), requires_grad=True)
        self.bone_lengths = Variable(bone_lengths_, requires_grad = False)
        self.lift_bone_directions = return_lift_bone_connections(self.bone_connections)

    def forward(self, pose_2d, R_drone, C_drone, pose3d_lift_directions, queue_index):
        #projection loss
        outputs = {}
        for loss in LOSSES:
            outputs[loss] = 0
        projected_2d, _ = take_bone_projection_pytorch(self.pose3d[queue_index, :, :].cpu(), R_drone, C_drone)
        outputs["proj"] = mse_loss(projected_2d, pose_2d)

        #smoothness
        #if (queue_index != self.window_size-1 and queue_index != 0):
        #    outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :] - self.pose3d[queue_index+1, :, :], self.pose3d[queue_index-1, :, :]- self.pose3d[queue_index, :, :])
        if (queue_index != self.window_size-1 and queue_index != 0):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :]) +  mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])
        elif (queue_index != self.window_size-1 ):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :])
        elif (queue_index != 0):
            outputs["smooth"] = mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])

        #bone length consistency 
        bonelosses = Variable(torch.zeros([self.NUM_OF_JOINTS-1,1]), requires_grad = False)
        for i, bone in enumerate(self.bone_connections):
            length_of_bone = (torch.sum(torch.pow(self.pose3d[queue_index, :, bone[0]] - self.pose3d[queue_index, :, bone[1]], 2)))
            bonelosses[i] = torch.pow((self.bone_lengths[i] - length_of_bone),2)
        outputs["bone"] = torch.sum(bonelosses)/bonelosses.data.nelement()

        pose_est_directions = torch.zeros([3, len(self.lift_bone_directions)])
        for i, bone in enumerate(self.lift_bone_directions):
            bone_vector = self.pose3d[queue_index, :, bone[0]] - self.pose3d[queue_index, :, bone[1]]
            pose_est_directions[:, i] = bone_vector/(torch.norm(bone_vector)+EPSILON)

        #hip_index = self.joint_names.index('spine1')
        #hip = self.pose3d[queue_index, :, hip_index].unsqueeze(1)
        #temp_pose3d_t = torch.sub(self.pose3d[queue_index, :, :], hip)
        #normalized_pose_3d, temp_pose3d_t = normalize_pose(self.pose3d[queue_index, :, :], self.joint_names, is_torch = True)


        #if (queue_index != self.window_size-1 and queue_index != 0):
        #    temp_pose3d_t_p_1 = torch.sub(self.pose3d[queue_index+1, :, :], self.pose3d[queue_index+1, :, hip_index].unsqueeze(1))
        #    temp_pose3d_t_m_1 = torch.sub(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index-1, :, hip_index].unsqueeze(1))
        #    outputs["smoothpose"] = mse_loss(temp_pose3d_t, temp_pose3d_t_p_1) +  mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)
        #elif (queue_index != self.window_size-1 ):
        #    temp_pose3d_t_p_1 = torch.sub(self.pose3d[queue_index+1, :, :], self.pose3d[queue_index+1, :, hip_index].unsqueeze(1))
        #    outputs["smoothpose"] = mse_loss(temp_pose3d_t, temp_pose3d_t_p_1)
        #elif (queue_index != 0):
        #    temp_pose3d_t_m_1 = torch.sub(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index-1, :, hip_index].unsqueeze(1))
        #    outputs["smoothpose"] = mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)

        #normalize pose
        #max_z = torch.max(temp_pose3d_t[2,:])
        #min_z = torch.min(temp_pose3d_t[2,:])
        #normalized_pose_3d = (pose3d_lift)/(max_z - min_z)
        outputs["lift"]= mse_loss(pose_est_directions, pose3d_lift_directions)
        return outputs
    
    def init_pose3d(self, pose3d_, queue_index):
        self.pose3d.data[queue_index, :, :] = pose3d_.data[:]


class toy_example(torch.nn.Module):
    def __init__(self, model, loss_dict, weights, data_list, M):
        super(toy_example, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([4,1]), requires_grad=True)

    def forward(self):
        return self.pose3d[3]**2 + self.pose3d[2]*self.pose3d[1]*self.pose3d[0] + self.pose3d[0]**4
 
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_calibration(torch.nn.Module):

    def __init__(self, model, loss_dict, weights, data_list, M):
        super(pose3d_calibration, self).__init__()
        self.bone_connections, _, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.left_bone_connections, self.right_bone_connections, _ = split_bone_connections(self.bone_connections)
        self.pose3d = torch.nn.Parameter(torch.zeros([3, self.NUM_OF_JOINTS]), requires_grad=True)
        self.energy_weights = weights
        self.loss_dict = loss_dict
        self.data_list = data_list
        self.M = torch.from_numpy(M).float()
    
    def forward(self):        
        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        bonelosses = Variable(torch.zeros([len(self.left_bone_connections),]), requires_grad = False)
        for i, l_bone in enumerate(self.left_bone_connections):
            r_bone = self.right_bone_connections[i]
            left_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, l_bone[0]] - self.pose3d[:, l_bone[1]], 2)))
            right_length_of_bone = (torch.sum(torch.pow(self.pose3d[:, r_bone[0]] - self.pose3d[:, r_bone[1]], 2)))
            bonelosses[i] = torch.pow((left_length_of_bone - right_length_of_bone),2)
        output["sym"] += torch.sum(bonelosses)/bonelosses.data.nelement()
        #residuals = bonelosses* self.energy_weights["sym"]

        #pose_3d_M = torch.mm(self.pose3d, self.M)
        pose_3d_M = self.pose3d.cpu()
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            R_drone_torch = torch.from_numpy(R_drone_).float()
            C_drone_torch = torch.from_numpy(C_drone_).float()
            bone_2d_torch = torch.from_numpy(bone_2d_).float()
            projected_2d, _ = take_bone_projection_pytorch(pose_3d_M, R_drone_torch, C_drone_torch)
            output["proj"] += mse_loss(projected_2d, bone_2d_torch)
            #current_residuals = find_residuals(projected_2d, bone_2d_torch)* self.energy_weights["proj"]
            #residuals = torch.cat((residuals, current_residuals))

        overall_output = Variable(torch.FloatTensor([0]))
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]
        return overall_output

    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_flight(torch.nn.Module):

    def __init__(self, model, bone_lengths, window_size, loss_dict, weights, data_list, lift_list, M):
        super(pose3d_flight, self).__init__()
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.window_size = window_size
        self.pose3d = torch.nn.Parameter(torch.zeros([self.window_size+1, 3, self.NUM_OF_JOINTS]), requires_grad=True)
        self.bone_lengths = Variable(bone_lengths, requires_grad = False)
        self.loss_dict = loss_dict
        self.data_list = data_list
        self.lift_list = lift_list
        self.energy_weights = weights
        self.lift_bone_directions = return_lift_bone_connections(self.bone_connections)
        self.M = torch.from_numpy(M).float()

    def forward_old(self):
        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        queue_index = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            #pose_3d_M = torch.mm(self.pose3d[queue_index, :, :].cpu(), self.M)
            pose_3d_M = self.pose3d[queue_index, :, :].cpu()
            #projection
            R_drone_torch = torch.from_numpy(R_drone_).float()
            C_drone_torch = torch.from_numpy(C_drone_).float()
            bone_2d_torch = torch.from_numpy(bone_2d_).float()
            projected_2d, _ = take_bone_projection_pytorch(pose_3d_M, R_drone_torch, C_drone_torch)
            output["proj"] += mse_loss(projected_2d, bone_2d_torch)
            #current_residuals = find_residuals(projected_2d, bone_2d_torch)* self.energy_weights["proj"]
            #if (queue_index == 0):
            #    residuals = current_residuals
            #else: 
                #residuals =  torch.cat((residuals, current_residuals))

            #smoothness
           # if (queue_index != self.window_size-1 and queue_index != 0):
              #  output["smooth"] += mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :]) +  mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])
                #current_residuals = (find_residuals(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :]) + find_residuals(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :]))* self.energy_weights["smooth"]
           # elif (queue_index != self.window_size-1 ):
             #   output["smooth"] += mse_loss(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :])
                #current_residuals = find_residuals(self.pose3d[queue_index, :, :], self.pose3d[queue_index+1, :, :]) * self.energy_weights["smooth"]
           # elif (queue_index != 0):
               # output["smooth"] += mse_loss(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :])
                #current_residuals = find_residuals(self.pose3d[queue_index-1, :, :], self.pose3d[queue_index, :, :]) * self.energy_weights["smooth"]
            #residuals =  torch.cat((residuals, current_residuals))
            
            pose_vel = torch.zeros([2, 3, self.NUM_OF_JOINTS])
            if (queue_index != 0 or queue_index != 1):
                pose_vel[0]  = self.pose3d[queue_index-1, :, :]- self.pose3d[queue_index-2, :, :]
                pose_vel[1]  = self.pose3d[queue_index, :, :]- self.pose3d[queue_index-1, :, :]
                output["smooth"] += mse_loss(pose_vel[0] , pose_vel[1])


            #bone length consistency 
            bonelosses = Variable(torch.zeros([self.NUM_OF_JOINTS-1,]), requires_grad = False)
            for i, bone in enumerate(self.bone_connections):
                length_of_bone = (torch.sum(torch.pow(pose_3d_M[:,bone[0]] - pose_3d_M[:,bone[1]], 2)))
                bonelosses[i] = torch.pow((self.bone_lengths[i] - length_of_bone),2)
            output["bone"] += torch.sum(bonelosses)/(self.NUM_OF_JOINTS-1)
            #current_residuals = bonelosses* self.energy_weights["bone"]
            #residuals =  torch.cat((residuals, current_residuals))

            #lift
            pose3d_lift_directions = torch.from_numpy(self.lift_list[queue_index]).float()
            pose_est_directions = torch.zeros([3, len(self.lift_bone_directions)])
            for i, bone in enumerate(self.lift_bone_directions):
                bone_vector = pose_3d_M[:,bone[0]] - pose_3d_M[:,bone[1]]
                pose_est_directions[:, i] = bone_vector/(torch.norm(bone_vector)+EPSILON)
            output["lift"] += mse_loss(pose3d_lift_directions, pose_est_directions)
            #current_residuals = find_residuals(pose3d_lift_directions, pose_est_directions) * self.energy_weights["lift"]
            #residuals =  torch.cat((residuals, current_residuals))

            queue_index += 1

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]

        return overall_output

    
    def forward(self):
        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        queue_index = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            #pose_3d_curr = torch.mm(self.pose3d[queue_index, :, :].cpu(), self.M)
            pose_3d_curr = self.pose3d[queue_index, :, :].cpu()

            if (queue_index != 0): #future term does not have projection or lift term
                #projection
                R_drone_torch = torch.from_numpy(R_drone_).float()
                C_drone_torch = torch.from_numpy(C_drone_).float()
                bone_2d_torch = torch.from_numpy(bone_2d_).float()
                projected_2d, _ = take_bone_projection_pytorch(pose_3d_curr, R_drone_torch, C_drone_torch)
                output["proj"] += mse_loss(projected_2d, bone_2d_torch)

                #lift
                pose3d_lift_directions = torch.from_numpy(self.lift_list[queue_index]).float()
                pose_est_directions = torch.zeros([3, len(self.lift_bone_directions)])
                for i, bone in enumerate(self.lift_bone_directions):
                    bone_vector = pose_3d_curr[:,bone[0]] - pose_3d_curr[:,bone[1]]
                    pose_est_directions[:, i] = bone_vector/(torch.norm(bone_vector)+EPSILON)
                output["lift"] += mse_loss(pose3d_lift_directions, pose_est_directions)

            #smoothness term
            pose_vel = torch.zeros([2, 3, self.NUM_OF_JOINTS])
            if (queue_index != 0 or queue_index != 1):
                pose_vel[0]  = self.pose3d[queue_index-1, :, :]- self.pose3d[queue_index-2, :, :]
                pose_vel[1]  = self.pose3d[queue_index, :, :]- self.pose3d[queue_index-1, :, :]
                output["smooth"] += mse_loss(pose_vel[0] , pose_vel[1])

            #bone length consistency 
            bonelosses = Variable(torch.zeros([self.NUM_OF_JOINTS-1,]), requires_grad = False)
            for i, bone in enumerate(self.bone_connections):
                length_of_bone = (torch.sum(torch.pow(pose_3d_curr[:,bone[0]] - pose_3d_curr[:,bone[1]], 2)))
                bonelosses[i] = torch.pow((self.bone_lengths[i] - length_of_bone),2)
            output["bone"] += torch.sum(bonelosses)/(self.NUM_OF_JOINTS-1)

            queue_index += 1

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]

        return overall_output
    
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

class pose3d_future(torch.nn.Module):

    def __init__(self, model, pose3d_est_future, bone_lengths, R_drone, C_drone, loss_dict):
        super(pose3d_future, self).__init__()
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.future_pose3d = torch.nn.Parameter(torch.zeros([3, self.NUM_OF_JOINTS]), requires_grad=True)
        self.bone_lengths = Variable(bone_lengths, requires_grad = False)
        self.loss_dict = loss_dict
        self.R_drone = torch.from_numpy(R_drone).float()
        self.C_drone = torch.from_numpy(C_drone).float()
        pose3d_est_future_torch = torch.from_numpy(pose3d_est_future).float()
        self.projected_est, _ = take_bone_projection_pytorch(pose3d_est_future_torch, R_drone, C_drone)
    
    def forward(self):
        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        #projection
        projected_2d, _ = take_bone_projection_pytorch(self.future_pose3d, self.R_drone, self.C_drone)
        output["proj"] += mse_loss(projected_2d, self.projected_est)

        #bone length consistency 
        bonelosses = Variable(torch.zeros([self.NUM_OF_JOINTS-1,]), requires_grad = False)
        for i, bone in enumerate(self.bone_connections):
            length_of_bone = (torch.sum(torch.pow(self.future_pose3d[:,bone[0]] - self.future_pose3d[:,bone[1]], 2)))
            bonelosses[i] = torch.pow((self.bone_lengths[i] - length_of_bone),2)
        output["bone"] += torch.sum(bonelosses)/(self.NUM_OF_JOINTS-1)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]

        return overall_output
    
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.future_pose3d.data[:] = pose3d_.data[:]

