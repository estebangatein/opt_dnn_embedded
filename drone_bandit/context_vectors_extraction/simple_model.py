import torch.nn as nn

class simple_model(nn.Module):
    def __init__(self, img_channels=1, output_dim_objects=4):
        super(simple_model, self).__init__()

        # assumed dimensions of the input image: batch_size x 1 x 200 x 200
        self.conv1_block = nn.Sequential(nn.Conv2d(img_channels, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.exit1_block = nn.Sequential(nn.Linear(32 * 50 * 50, output_dim_objects))

        self.conv2_block = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.exit2_block = nn.Sequential(nn.Linear(64 * 13 * 13, output_dim_objects))
     

    def forward(self, x):
        x = self.conv1_block(x)
        early_exit = x.view(x.size(0), -1) # same as nn.Flatten
        early_exit = self.exit1_block(early_exit)

        x = self.conv2_block(x)
        x = x.view(x.size(0), -1)
        final_exit = self.exit2_block(x)
        
        return early_exit, final_exit


