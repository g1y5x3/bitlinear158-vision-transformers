import torch, random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader

from datasets import build_dataset
from models.backbone import ResNetBackbone
from models.transformer import TransformerBitLinear, Transformer
from models.detr import DETR, SetCriterion
from models.matcher import HungarianMatcher
from util.misc import collate_fn
from util.misc import rescale_bboxes, plot_results

# keep random seed fixed to keep my sanity
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Args:
  coco_path = "/workspace/coco"
  dataset_file = "coco"
  masks = False
args = Args()

dataset_train = build_dataset(image_set='train', args=args)
sampler_train = SequentialSampler(dataset_train)
batch_sampler_train = BatchSampler(sampler_train, batch_size=16, drop_last=True)
data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=collate_fn, num_workers=1)
batch_fetcher = iter(data_loader_train)

x, x_mask, y = next(batch_fetcher)

writer = SummaryWriter()

# DETR Model
device = "cuda"

backbone = ResNetBackbone()
transformer = TransformerBitLinear(256, 8, 6, 6, 2048, 0.1)
model = DETR(backbone, transformer, num_classes=91, num_queries=100).to(device)

matcher = HungarianMatcher()
criterion = SetCriterion(91, matcher).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()

model = model.train()
x, x_mask = x.to(device), x_mask.to(device) 
y = [{k: v.to(device) for k, v in t.items()} for t in y]

writer.add_graph(model, x)

for epochs in tqdm(range(500), desc="Training Progress"):
  optimizer.zero_grad()
  with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    outputs_logits, outputs_boxes = model(x, x_mask)
    loss = criterion(outputs_logits, outputs_boxes, y)

  writer.add_scalar("Loss/train", loss, epochs)
 
  scaler.scale(loss).backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
  scaler.step(optimizer)
  scaler.update()

#with torch.profiler.profile(
#    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#    record_shapes=True, profile_memory=True
#) as prof:
#    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
#        outputs = model(x, x_mask)
#        loss = criterion(outputs, y)
#
## Print profiler results
#print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=-1))

## Inspect predictions
#i = 3
#with torch.no_grad():
#  h, w = y[i]["size"]
#  im = x[i].cpu().numpy().transpose((1,2,0))
#  prob = outputs['pred_logits'][i,:].softmax(-1)[:, :-1]
#  keep = prob.max(-1).values > 0.25
#  plot_results(im, prob[keep], rescale_bboxes(outputs["pred_boxes"][i][keep,:], (w,h)))
#
## Inspect ground truth
#i = 3
#h, w = y[i]["size"]
#im = x[i].cpu().numpy().transpose((1,2,0))
#prob = torch.zeros((len(y[i]["labels"]), 91))
#prob[torch.arange(len(y[i]["labels"])), y[i]["labels"].cpu()] = 1  
#plot_results(im, prob, rescale_bboxes(y[i]["boxes"], (w,h)))