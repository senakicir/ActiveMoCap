import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Dropout


class LiftNet(nn.Module):
    def __init__(self, nb_channels, num_joints):
        super(LiftNet, self).__init__()
        self.conv1 = Conv2d(nb_channels, 36, 9)
        self.pool1 = MaxPool2d(2, 2)

        self.conv2 = Conv2d(36, 72, 5)
        self.pool2 = MaxPool2d(2, 2)

        self.conv3 = Conv2d(72, 72, 5)
        self.pool3 = MaxPool2d(2, 2)

        self.lin1 = Linear(7200, 1024)
        self.lin2 = Linear(1024, 2048)
        self.lin3 = Linear(2048, num_joints * 3)

        self.relu = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=False)
        self.dropout = Dropout(inplace=True, p=0.3)

    def forward(self, input_layer):
        out = self.conv1(input_layer)
        out = self.pool1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.pool2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.pool3(out)
        out = self.relu(out)
        out_shape = out.size()
        out = out.view(-1, out_shape[1]*out_shape[2]*out_shape[3])
        out = self.lin1(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.lin3(out)

        return out
