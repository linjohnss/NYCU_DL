import torch.nn as nn
class EEGNet(nn.Module):
    def __init__(self, activation=None):
        super(EEGNet, self).__init__()
        # input shape: (batch_size, 1, 2, 750)
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1,1), padding=(0,25), bias=False), # (batch_size, 16, 2, 750)
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False), # (batch_size, 32, 1, 750)
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.get_activation(activation),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0), # (batch_size, 32, 1, 187)
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1), padding=(0, 7), bias=False), # (batch_size, 32, 1, 187)
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.get_activation(activation),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0), # (batch_size, 32, 1, 23)
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            # in_features: 32 * 1 * 23 = 736
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x
    
    def get_activation(self, activation):
        if activation == 'ReLU':
            return nn.ReLU()
        elif activation == 'LeakyReLU':
            return nn.LeakyReLU()
        elif activation == 'ELU':
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")