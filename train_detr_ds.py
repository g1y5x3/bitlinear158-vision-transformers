import os, torch, deepspeed, random, argparse 
import numpy as np

import dataset.transforms as T
from model.backbone import ResNetBackbone
from model.transformer import TransformerBitLinear
from model.detr import DETR, SetCriterion
from model.matcher import HungarianMatcher
from dataset.coco import CocoDetection, collate_fn

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
	parser.add_argument('--num_classes', default=91, type=int)
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
	deepspeed.init_distributed()

	ds_config = {
  	"train_batch_size": args.batch_size,
  	"optimizer": {
  	  "type": "Adam",
  	  "params": {
  	    "lr": 1e-5
  	  }
  	},
  	"fp16": {
  	  "enabled": True,
			"auto_cast": True,
			"loss_scale": 0,
			"initial_scale_power": 11,
  	},
  	"zero_optimization": {
			"stage": 1,
			"reduce_bucket_size": 5e8,
		},
		"tensorboard": {
			"enabled": True,
			"output_path": "run/",
			"job_name": "train_detr_bitlinear_pred",
		},
		"comms_logger": {
			"enabled": True,
			"verbose": False,
			"prof_all": True,
			"debug": False
		}
	}

	rank = int(os.getenv("LOCAL_RANK", "0"))

	seed = args.seed + rank
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
 
	# TODO: replace with Albumentation
	scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
	transform_train = T.Compose([
	  # augumentation
	  T.RandomHorizontalFlip(),
	  T.RandomSelect(
	    T.RandomResize(scales, max_size=1333),
	    T.Compose([
	      T.RandomResize([400, 500, 600]),
	      T.RandomSizeCrop(384, 600),
	      T.RandomResize(scales, max_size=1333),
	    ])
	  ),
	  # normalize
	  T.Compose([
	    T.ToTensor(),
	    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	  ])
	])  

	# avoid downloading the same file at the same time during model initialization
	if rank != 0: torch.distributed.barrier()

	dataset_train = CocoDetection(args.coco_path + "/train2017", args.coco_path + "/annotations/instances_train2017.json", transform_train)

	# TODO: use a ViT based backbone
	backbone = ResNetBackbone()
	transformer = TransformerBitLinear(args.hidden_dim, args.nheads, args.enc_layers, args.dec_layers, args.dim_feedforward, args.dropout)
	
	model = DETR(backbone=backbone, transformer=transformer, num_classes=args.num_classes, num_queries=args.num_queries)

	matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)

	# TODO: let criterion return all 3 losses and apply the weight at the end before backprop
	criterion = SetCriterion(args.num_classes, matcher, args.eos_coef, weight=(args.dice_loss_coef, args.bbox_loss_coef, args.giou_loss_coef))

	if rank == 0: torch.distributed.barrier()

	# TODO: calculate the number of tenary parameters for transformers
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('number of params:', n_parameters)

	# deepseed initialization
	model_engine, optimizer, data_loader_train, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), 
																																			 training_data=dataset_train, collate_fn=collate_fn,	
																																			 config=ds_config)
	device = torch.device("cuda", rank)
	model.to(device)
	criterion.to(device)

	for epoch in range(args.start_epoch, args.epochs):
		model.train()
		criterion.train()
		for samples, masks, targets in data_loader_train:
			optimizer.zero_grad()

			samples = samples.to(device)
			masks 	= masks.to(device)
			# TODO: separate them into targets_class and target_bboxes
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

			with torch.autocast(device_type="cuda", dtype=torch.float16):
				outputs_logits, outputs_boxes = model_engine(samples, masks)
				loss = criterion(outputs_logits, outputs_boxes, targets)

			model_engine.backward(loss)
			model_engine.step()

		# TODO: add eval on rank=0

if __name__ == '__main__':
	parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	main(args)