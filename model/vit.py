import torch
from torch import nn
from transformer import ViTEncoderLayer

class ViT(nn.modules):
  def __init__(self, image_size=224, channels=3, patch_size=16, d_model=768):
    super().__init__()
    # assume images to be square
    num_patches = (image_size // patch_size)**2
    patch_dim = channels * (patch_size**2)
    self.position_embedding = nn.Parameter(torch.randn(1, num_patches+1, d_model))