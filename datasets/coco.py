import torch
import torch.utils.data
import torchvision

from pathlib import Path
from pycocotools import mask as coco_mask

import datasets.transforms as T

from torchvision.transforms import v2


class CocoDetection(torchvision.datasets.CocoDetection):
  def __init__(self, img_folder, ann_file, transforms, return_masks):
    super(CocoDetection, self).__init__(img_folder, ann_file)
    self._transforms = transforms
    self.prepare = ConvertCocoPolysToMask(return_masks)

  def __getitem__(self, idx):
    img, target = super(CocoDetection, self).__getitem__(idx)
    image_id = self.ids[idx]
    target = {'image_id': image_id, 'annotations': target}
    img, target = self.prepare(img, target)
    if self._transforms is not None:
      img, target = self._transforms(img, target)
    return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def make_coco_transforms(image_set):

    normalize = T.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
