#import datetime
#import json
#import time
#from torch.utils.data import DataLoader, DistributedSampler
#import datasets
#
#import torch, random, argparse
#import numpy as np

import os, torch, deepspeed, time, datetime, random, argparse 
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

import util.misc as utils
from models.backbone import ResNetBackbone
from models.transformer import Transformer, TransformerBitLinear
from models.detr import DETR, SetCriterion
from models.matcher import HungarianMatcher
from datasets import build_dataset
from engine import train_one_epoch

def get_args_parser():
	parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
	parser.add_argument('--lr', default=1e-4, type=float)
	parser.add_argument('--lr_backbone', default=1e-5, type=float)
	parser.add_argument('--batch_size', default=16, type=int)
	parser.add_argument('--weight_decay', default=1e-4, type=float)
	parser.add_argument('--epochs', default=300, type=int)
	parser.add_argument('--lr_drop', default=200, type=int)
	parser.add_argument('--clip_max_norm', default=0.1, type=float,
											help='gradient clipping max norm')
	# Backbone
	parser.add_argument('--backbone', default='resnet50', type=str,
											help="Name of the convolutional backbone to use")
	# Transformer
	parser.add_argument('--enc_layers', default=6, type=int,
											help="Number of encoding layers in the transformer")
	parser.add_argument('--dec_layers', default=6, type=int,
											help="Number of decoding layers in the transformer")
	parser.add_argument('--dim_feedforward', default=2048, type=int,
											help="Intermediate size of the feedforward layers in the transformer blocks")
	parser.add_argument('--hidden_dim', default=256, type=int,
											help="Size of the embeddings (dimension of the transformer)")
	parser.add_argument('--dropout', default=0.1, type=float,
											help="Dropout applied in the transformer")
	parser.add_argument('--nheads', default=8, type=int,
											help="Number of attention heads inside the transformer's attentions")
	parser.add_argument('--num_queries', default=100, type=int,
											help="Number of query slots")
	# Matcher
	parser.add_argument('--cost_class', default=1, type=float,
											help="Class coefficient in the matching cost")
	parser.add_argument('--cost_bbox', default=5, type=float,
											help="L1 box coefficient in the matching cost")
	parser.add_argument('--cost_giou', default=2, type=float,
											help="giou box coefficient in the matching cost")
	# Loss coefficients
	parser.add_argument('--dice_loss_coef', default=1, type=float)
	parser.add_argument('--bbox_loss_coef', default=5, type=float)
	parser.add_argument('--giou_loss_coef', default=2, type=float)
	parser.add_argument('--eos_coef', default=0.1, type=float,
											help="Relative classification weight of the no-object class")
	# Dataset parameters
	parser.add_argument('--dataset_file', default='coco')
	parser.add_argument('--coco_path', type=str)
	# Others
	parser.add_argument('--output_dir', default='',
											help='path where to save, empty for no saving')
	parser.add_argument('--device', default='cuda',
											help='device to use for training / testing')
	parser.add_argument('--seed', default=42, type=int)
	parser.add_argument('--resume', default='', help='resume from checkpoint')
	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
											help='start epoch')
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--num_workers', default=2, type=int)
	# Distributed training parameters
	parser.add_argument('--world_size', default=1, type=int,
											help='number of distributed processes')
	parser.add_argument('--local_rank', default=-1, type=int,
										  help='parameters used by deepspeed')
  # Segmentation
	parser.add_argument('--masks', action='store_true',
											help="Train segmentation head if the flag is provided")
	return parser


def main(args):
	rank = int(os.getenv("LOCAL_RANK", "0"))
	# world_size = int(os.getenv("WORLD_SIZE", "1"))
	device = torch.device("cuda", rank)

	ds_config = {
  	"train_batch_size": 2,
  	"optimizer": {
  	  "type": "Adam",
  	  "params": {
  	    "lr": 1e-5
  	  }
  	},
  	"fp16": {
  	  "enabled": True
  	},
  	"zero_optimization": {
			"stage": 0,
		}
	}

  # fix the seed for reproducibility (and your sanity)
	seed = args.seed + rank
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
    
	# model and criterion
	backbone = ResNetBackbone()		# currently not using the args
	transformer = TransformerBitLinear(args.hidden_dim, args.nheads, args.enc_layers, args.dec_layers, args.dim_feedforward, args.dropout)
	model = DETR(backbone=backbone, transformer=transformer, num_classes=91, num_queries=args.num_queries)
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('number of params:', n_parameters)

	matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)
	criterion = SetCriterion(91, matcher, args.eos_coef, (args.dice_loss_coef, args.bbox_loss_coef, args.giou_loss_coef))

	model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
	print(f"Rank {rank} Device {device.type} - ZeRO Stage: {model_engine.zero_optimization_stage()}")

	# # create dataset
	# dataset_train = build_dataset(image_set='train', args=args)

	# # dataset_val   = build_dataset(image_set='val', args=args)
	# sampler_train = torch.utils.data.RandomSampler(dataset_train)
	# # sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
	# batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

	# data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, 
	# 															 collate_fn=utils.collate_fn, num_workers=args.num_workers)
	# # data_loader_val   = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
	# # 														   collate_fn=utils.collate_fn, num_workers=args.num_workers)

	# print("Start training")
	# start_time = time.time()

	# for epoch in range(args.start_epoch, args.epochs):
	# 	model.train()
	# 	criterion.train()

	# 	with tqdm(data_loader_train, unit="batch") as tepoch: 
	# 		for samples, masks, targets in tepoch:
	# 			samples = samples.to(device)
	# 			masks = masks.to(device)
	# 			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

	# 			optimizer.zero_grad()

	# 			outputs_logits, outputs_boxes = model_engine(samples, masks)
	# 			loss = criterion(outputs_logits, outputs_boxes, targets)

	# 			model_engine.backward(loss)
	# 			model_engine.step()

	# 			# tepoch.set_postfix(loss=loss.item())		

	# total_time = time.time() - start_time
	# total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	# print('Training time {}'.format(total_time_str))		

if __name__ == '__main__':
	parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	main(args)