import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation

import transforms as T
import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict
import cv2

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset_image
    ds = ReferDataset_image(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    save_annot_dir = 'crop_annot_train'
    save_image_dir = 'crop_image_train'
    if not os.path.exists(save_annot_dir):
        os.mkdir(save_annot_dir)
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)


    # housekeeping
    start_time = time.time()

    # resume training (optimizer, lr scheduler, and the epoch)

    for idx,data in enumerate(data_loader):
        image, target = data
        image = image.squeeze()
        target = target.squeeze()
        h,w = target.shape


        position = np.where(target==1)
        y1, y2 = np.min(position[0]),np.max(position[0])
        x1, x2 = np.min(position[1]),np.max(position[1])
        extend_h = max(10,int((y2-y1)/10))
        extend_w = max(10,int((x2-x1)/10))
        extend_x1 = max(0,x1-extend_w)
        extend_x2 = min(w,x2+extend_w)

        extend_y1 = max(0,y1-extend_h)
        extend_y2 = min(h,y2+extend_h)
        crop_image = np.array(image[extend_y1:extend_y2,extend_x1:extend_x2])
        crop_anno = np.array(target[extend_y1:extend_y2,extend_x1:extend_x2])*255

        cv2.imwrite(os.path.join(save_image_dir,'{}.png'.format(idx)),crop_image)
        cv2.imwrite(os.path.join(save_annot_dir,'{}.png'.format(idx)),crop_anno)

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
