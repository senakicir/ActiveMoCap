import torch as torch
from torch.autograd import grad, backward
import numpy as np

class toy_example(torch.nn.Module):
    def __init__(self):
        super(toy_example, self).__init__()
        self.pose3d = torch.nn.Parameter(torch.zeros([4,1]), requires_grad=True)

    def forward(self):
        return self.pose3d[3]**2 + self.pose3d[2]*self.pose3d[1]*self.pose3d[0] + self.pose3d[0]**4
 
    def init_pose3d(self, pose3d_np):
        pose3d_ = torch.from_numpy(pose3d_np).float()
        self.pose3d.data[:] = pose3d_.data[:]

def fun_hessian_grad(pytorch_objective, x):
    hess_size = x.size
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

def fun_hessian_backward(pytorch_objective, x):
    #first derivative
    pytorch_objective.zero_grad()
    pytorch_objective.init_pose3d(x)
    overall_output = pytorch_objective.forward()
    overall_output.backward(create_graph=True, retain_graph = True)
    gradient_torch = pytorch_objective.pose3d.grad
    #second derivative
    gradient_torch.backward(pytorch_objective.pose3d)
    hessian_torch =  pytorch_objective.pose3d.grad
    hessian = hessian_torch.detach().numpy()
    return hessian

if __name__ == "__main__":
    objective = toy_example()
    x = np.ones([4,1])
    hessian_grad = fun_hessian_grad(objective, x)
    print("using grad:", hessian_grad)

    hessian_backward = fun_hessian_backward(objective, x)
    print("using backward:", hessian_backward)
