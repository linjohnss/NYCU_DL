import torch
from torch import nn
from diffusers import UNet2DModel
class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=24, class_emb_size=3):
    super().__init__()
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.label_emb = nn.Sequential(
            nn.Linear(num_classes, 64*64),
            nn.ReLU(),
    )

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=64,           # the target image resolution
        in_channels=3 + 24,           # Additional input channels for class cond.
        out_channels=3,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        # block_out_channels=(128, 128, 256, 256, 512, 512), # the ResNet channels per block
        block_out_channels=(64, 64, 128, 128, 256, 256), # the ResNet channels per block
        down_block_types=( 
            "DownBlock2D",        # a regular ResNet downsampling block
            "DownBlock2D",        # a regular ResNet downsampling block
            "DownBlock2D",        # a regular ResNet downsampling block
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",
            "DownBlock2D",        # a regular ResNet downsampling block
        ), 
        up_block_types=(
            "UpBlock2D",          # a regular ResNet upsampling block
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
            "UpBlock2D",          # a regular ResNet upsampling block
            "UpBlock2D",          # a regular ResNet upsampling block
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
    # class_cond = self.label_emb(class_labels).view(bs, 1, w, h)
    class_cond = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(bs, class_labels.shape[1], w, h)
    net_input = torch.cat((x, class_cond), 1)
    # net_input = x + class_cond
    return self.model(net_input, t).sample   