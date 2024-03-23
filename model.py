
import torch.nn as nn
import torch.nn.functional as F
class Conv2D_Voice_Model(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=6):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=(1, 80), stride=(1, stride))
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d((1, 4))
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=(1, 3))
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d((1, 4))
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(1, 3))
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d((1, 4))
        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=(1, 3))
        self.bn4 = nn.BatchNorm2d(2 * n_channel)
        self.pool4 = nn.MaxPool2d((1, 4))
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = F.avg_pool2d(x, (1, x.shape[-1]))
        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=-2)
        x = self.fc1(x)

        return x

