import torch
import torch.nn as nn

class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()

        self.block_0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

        self.block_3 = nn.Sequential(
            nn.Linear(in_features=320000, out_features=4, bias=True)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.block_6 = nn.Sequential(
            nn.Linear(in_features=320000, out_features=4, bias=True)
        )

        self.block_7 = nn.Sequential(
            nn.Linear(in_features=320000, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2, bias=True)
        )

        self.block_8 = nn.Sequential(
            nn.Linear(in_features=320000, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1, bias=True)
        )

    def forward(self, x):
        out_0 = self.block_0(x)
        out_1 = self.block_1(out_0)
        out_2 = out_1.flatten(start_dim=1, end_dim=-1)
        out_3 = self.block_3(out_2)
        out_4 = self.block_4(out_0)
        out_5 = out_4.flatten(start_dim=1, end_dim=-1)
        out_6 = self.block_6(out_5)
        out_7 = self.block_7(out_5)
        out_8 = self.block_8(out_5)
        return out_3, out_6, out_7, out_8
