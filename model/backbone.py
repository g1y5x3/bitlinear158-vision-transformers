import torch
import torchvision
import torch.nn.functional as F

from torch import nn, Tensor
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter

# TODO: add other backbone options
class ResNetBackbone(nn.Module):
  def __init__(self):
    super().__init__()
    # NOTE: FrozenBatchNorm2d works but it increases the memory usage of `aten::conv2d` when autocast to float16 compare to float32, probably made 
    # some extra copy somewhere. Also, default BatchNorm2d is slightly faster.
    # resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
    resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # TODO: get other layers' output as features for the aux loss function
    self.resnet = IntermediateLayerGetter(resnet50, return_layers={"layer4": "0"})
    self.num_channels = 2048

  def forward(self, x: Tensor, mask: Tensor):
    feature = self.resnet(x)["0"] 
    feature_mask = F.interpolate(mask[None].to(x.dtype), size=feature.shape[-2:]).to(torch.bool)[0]
    return feature, feature_mask