import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoEE(nn.Module):
    def __init__(self):
        super(TwoEE, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ee_branch_binary = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.ee_branch_binary_bn = nn.BatchNorm2d(128)
        self.ee_branch_binary_fc = nn.Linear(128 * 50 * 50, 1)

        self.ee_branch_reg = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.ee_branch_reg_bn = nn.BatchNorm2d(128)
        self.ee_branch_reg_fc = nn.Linear(128 * 50 * 50, 1)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.binary_fc1 = nn.Linear(512 * 25 * 25, 256)
        self.binary_fc2 = nn.Linear(256, 1)

        self.regression_fc1 = nn.Linear(512 * 25 * 25, 256)
        self.regression_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        ee_branch_binary = F.relu(self.ee_branch_binary_bn(self.ee_branch_binary(x)))
        ee_branch_binary = torch.flatten(ee_branch_binary, 1)
        ee_branch_binary = self.ee_branch_binary_fc(ee_branch_binary)

        ee_branch_reg = F.relu(self.ee_branch_reg_bn(self.ee_branch_reg(x)))
        ee_branch_reg = torch.flatten(ee_branch_reg, 1)
        ee_branch_reg = self.ee_branch_reg_fc(ee_branch_reg)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = torch.flatten(x, 1)  

        binary_output = F.relu(self.binary_fc1(x))
        binary_output = self.binary_fc2(binary_output)

        regression_output = F.relu(self.regression_fc1(x))
        regression_output = self.regression_fc2(regression_output)

        return ee_branch_binary, ee_branch_reg, binary_output, regression_output


