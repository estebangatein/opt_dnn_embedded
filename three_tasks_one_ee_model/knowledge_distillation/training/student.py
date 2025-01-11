import torch
import torch.nn as nn
import torch.nn.functional as F



class StudentOneEE(nn.Module):
    def __init__(self):
        super(StudentOneEE, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # early exit for object detection
        self.obj_detect_fc_ee = nn.Linear(32 * 50 * 50, 4)

        # additional pooling layer to reduce dimensions
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.obj_detect_fc_final = nn.Linear(128 * 6 * 6, 4)

        self.binary_fc1 = nn.Linear(128 * 6 * 6, 64)
        self.binary_fc2 = nn.Linear(64, 1)

        self.regression_fc1 = nn.Linear(128 * 6 * 6, 64)
        self.regression_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        obj_detect_ee = x.view(-1, 32 * 50 * 50)
        obj_detect_ee = self.obj_detect_fc_ee(obj_detect_ee)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.pool5(x)

        x = x.flatten(1)

        obj_detect_final = self.obj_detect_fc_final(x)

        binary_output = F.relu(self.binary_fc1(x))
        binary_output = self.binary_fc2(binary_output)

        regression_output = F.relu(self.regression_fc1(x))
        regression_output = self.regression_fc2(regression_output)

        return obj_detect_ee, obj_detect_final, binary_output, regression_output