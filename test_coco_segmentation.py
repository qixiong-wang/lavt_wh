import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F
import mmcv


def get_dataset(image_set, transform, args):
    from data.dataset_coco import CocoDataset
    ds = CocoDataset(ann_file='annotations/instances_val2014.json',)
    num_classes = 80

    return ds, num_classes



def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    model_class = BertModel
    single_bert_model = model_class.from_pretrained(args.ck_bert)
    # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
    if args.ddp_trained_weights:
        single_bert_model.pooler = None
    single_bert_model.load_state_dict(checkpoint['bert_model'])
    bert_model = single_bert_model.to(device)

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables

    header = 'Test:'
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader_test, 100, header):
            image, target = data
            image, target= image.to(device), target.to(device)
            output = model(image)
            import pdb
            pdb.set_trace()

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
