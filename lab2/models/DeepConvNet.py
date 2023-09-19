import torch.nn as nn
class DeepConvNet(nn.Module):
    def __init__(self, C=2, T=750, N=2, activation=None):
        super(DeepConvNet, self).__init__()

        # Define the architecture based on the given table
        self.conv_layers = nn.Sequential(
            # input shape: (batch_size, 1, 2, 750)
            nn.Conv2d(1, 25, kernel_size = (1, 5)), # (batch_size, 25, 2, 746)
            nn.Conv2d(25, 25, kernel_size = (C, 1)), # (batch_size, 25, 1, 746)
            nn.BatchNorm2d(25, eps = 1e-5, momentum = 0.1),
            self.get_activation(activation), 
            nn.MaxPool2d(kernel_size = (1, 2)), # (batch_size, 25, 1, 373)
            nn.Dropout(p = 0.5),

            nn.Conv2d(25, 50, kernel_size = (1, 5)), # (batch_size, 50, 1, 369)
            nn.BatchNorm2d(50, eps = 1e-5, momentum = 0.1),
            self.get_activation(activation),
            nn.MaxPool2d(kernel_size = (1, 2)), # (batch_size, 50, 1, 184)
            nn.Dropout(p = 0.5),

            nn.Conv2d(50, 100, kernel_size = (1, 5)), # (batch_size, 100, 1, 180)
            nn.BatchNorm2d(100, eps = 1e-5, momentum = 0.1),
            self.get_activation(activation),
            nn.MaxPool2d(kernel_size = (1, 2)), # (batch_size, 100, 1, 90)
            nn.Dropout(p = 0.5),

            nn.Conv2d(100, 200, kernel_size = (1, 5)), # (batch_size, 200, 1, 86)
            nn.BatchNorm2d(200, eps = 1e-5, momentum = 0.1),
            self.get_activation(activation),
            nn.MaxPool2d(kernel_size = (1, 2)), # (batch_size, 200, 1, 43)
            nn.Dropout(p = 0.5),
        )

        self.classify = nn.Sequential(
            # in_features: 200 * 1 * 43 = 8600
            nn.Linear(in_features=8600, out_features=N, bias=True)
        )

    def forward(self, x):
        # x.shape = (batch_size, 1, 2, 750)
        x = self.conv_layers(x) # x.shape = (batch_size, 200, 1, 43)
        x = x.view(x.size(0), -1) # x.shape = (batch_size, 8600)
        x = self.classify(x) # x.shape = (batch_size, 2)
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