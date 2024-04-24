from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops import box_convert
from torchvision.ops import generalized_box_iou_loss as giou_loss
from models.position_encoding import PositionEmbeddingSine


# TODO: separate the encoder and decoder when applying to other DETR variant
class DETR(nn.Module):
  def __init__(self, backbone, transformer, num_classes, num_queries):
    super().__init__()

    num_channels = backbone.num_channels
    num_hidden = transformer.d_model

    self.backbone = backbone
    self.position_encoding = PositionEmbeddingSine(num_hidden//2, normalize=True)
    self.input_proj = nn.Conv2d(num_channels, num_hidden, kernel_size=1)
    self.query_embed = nn.Embedding(num_queries, num_hidden)
    self.transformer = transformer
    self.class_embed = nn.Linear(num_hidden, num_classes + 1)
    self.bbox_embed = nn.Sequential(
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(),
      nn.Linear(num_hidden, num_hidden),
      nn.ReLU(),
      nn.Linear(num_hidden, 4)
    )

  def forward(self, x: Tensor, mask: Tensor=None):
    bsz, _, h, w = x.shape

    if mask is None: mask = torch.zeros((bsz, h, w), dtype=torch.bool, device=x.device)
    
    # mask is interpolated after convolutions
    feature, feature_mask = self.backbone(x, mask)
    feature_embed = self.position_encoding(feature, feature_mask)

    # [N,C,H,W] -> [N,HxW,C]
    src = self.input_proj(feature).flatten(2).transpose(1,2)
    src_mask = feature_mask.flatten(1)
    src_embed = feature_embed.flatten(2).transpose(1,2)
    query_embed = self.query_embed.weight.unsqueeze(0).repeat(bsz,1,1)

    # TODO: fix the inversion of mask inside positional encoder
    output_embedding = self.transformer(src, ~src_mask, src_embed, query_embed)

    outputs_class = self.class_embed(output_embedding)
    outputs_coord = self.bbox_embed(output_embedding).sigmoid()
    outputs = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

    return outputs

class SetCriterion(nn.Module):
  def __init__(self, num_classes: int, matcher: nn.Module, eos_coef: float=0.1, weight: Tuple[float, float, float]=(1.0, 1.0, 1.0)) -> None:
    super().__init__()
    self.num_classes = num_classes
    self.matcher = matcher
    self.weight = weight
    empty_weight = torch.ones(self.num_classes + 1)
    empty_weight[-1] = eos_coef
    self.register_buffer('empty_weight', empty_weight)

  def forward(self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor]):
    # match predictions with ground truth and produce src and target tensors for computing losses
    # slowest part of the entire algorithm due to extra trips to the cpu inside matcher and targets having dynamic length
    indices = self.matcher(outputs, targets)
    batch_idx, query_idx, labels, target_boxes = [], [], [], []
    for i, ((src_idx, tgt_idx), target) in enumerate(zip(indices, targets)):
      batch_idx.append(torch.full_like(src_idx, i))
      query_idx.append(src_idx)
      labels.append(target["labels"][tgt_idx])
      target_boxes.append(target["boxes"][tgt_idx])
    batch_idx = torch.cat(batch_idx, dim=0)
    query_idx = torch.cat(query_idx, dim=0)
    labels = torch.cat(labels, dim=0)

    src_logits = outputs["pred_logits"]
    target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
    target_classes[batch_idx, query_idx] = labels
    loss_ce = F.cross_entropy(src_logits.transpose(1,2), target_classes, self.empty_weight, reduction="mean")

    src_boxes = outputs["pred_boxes"][batch_idx, query_idx]
    target_boxes = torch.cat(target_boxes)
    loss_l1 = F.l1_loss(src_boxes, target_boxes, reduction="mean")
    loss_giou = giou_loss(box_convert(src_boxes,"cxcywh","xyxy"), box_convert(target_boxes,"cxcywh","xyxy"), reduction="mean")

    loss = self.weight[0]*loss_ce + self.weight[1]*loss_l1 + self.weight[2]*loss_giou

    return loss 