import torch
import torch.nn as nn

class ShallowConvNet(nn.Module):
    def __init__(self):
        super(ShallowConvNet, self).__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 13), stride=(1, 1)),
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=(2, 1), stride=(1, 1)),
        )

        self.batch_norm = nn.BatchNorm2d(40, eps = 1e-5, momentum = 0.1)
        
        self.pooling = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))
        
        self.classification = nn.Sequential(
            nn.Linear(in_features=4040, out_features=2, bias=True)
        )
        
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = x ** 2
        x = self.pooling(x)
        x = torch.log(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        
        return x