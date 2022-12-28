import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import random

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation
import pdb
import transforms as T
import utils
from pringfile import log_string
import numpy as np
import shutil, glob
from transform1 import new_transform

import torch.nn.functional as F

import gc
from collections import OrderedDict

def copy_dirs(target_path):
    file_count = 0
    for file in glob.glob('*.py'):
        shutil.copy(file, target_path)
        file_count+=1
    print(file_count)

# def log_string(out_str):
#     LOG_FOUT.write(out_str+'\n')
#     LOG_FOUT.flush()
#     print(out_str)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
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


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)


def evaluate(model, data_loader, bert_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, sentences1, attentions = data
            image, target, sentences, sentences1, attentions = image.cuda(non_blocking=True), \
                                                               target.cuda(non_blocking=True), \
                                                               sentences.cuda(non_blocking=True), \
                                                               sentences1.cuda(non_blocking=True), \
                                                               attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            sentences1 = sentences1.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                embedding1 = embedding
                loss_contra, loss_lansim, output = model(image, embedding, embedding1, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    log_string('Final results:')
    log_string('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    log_string(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, sentences1, attentions = data
        image, target, sentences, sentences1, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True), \
                                               sentences1.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        sentences1 = sentences1.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            last_hidden_states1 = bert_model(sentences1, attention_mask=attentions)[0]  # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            embedding1 = last_hidden_states1.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            loss_contra, loss_lansim, output = model(image, embedding, embedding1, l_mask=attentions)
            # pdb.set_trace()
        else:
            output = model(image, sentences, l_mask=attentions)

        # pdb.set_trace()
        # target60 = torch.tensor(target, dtype=torch.float32)
        # target60 = torch.tensor(adp60(target60), dtype=torch.int64)
        # target120 = torch.tensor(target, dtype=torch.float32)
        # target120 = torch.tensor(adp120(target120), dtype=torch.int64)

        # loss_lansim = cossim(lanp, lanm, attentions)
        loss_seg = criterion(output, target)
        loss = loss_seg + loss_lansim * 0.01 + loss_contra * 0.01
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_seg=loss_seg.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_lansim=loss_lansim.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_contra=loss_contra.item(), lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(loss0=loss.item())
        # metric_logger.update(loss60=loss60.item())
        # metric_logger.update(loss120=loss120.item())

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(args):
    dataset, num_classes = get_dataset("train",
                                       new_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)

    # batch sampler
    log_string(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    # pdb.set_trace()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    # pdb.set_trace()
    log_string(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    # model = torch.nn.parallel.DataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if resume_flag:
        checkpoint = torch.load(cp_path, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # {"params": [p for p in single_model.contrastive.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # pdb.set_trace()
    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    # warm_up_iter = 800
    # lambda0 = lambda x: x / warm_up_iter if x < warm_up_iter else \
    #         (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    #
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda0)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if resume_flag:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
        if os.path.exists(os.path.join(log_dir, 'best_performance.npy')):
            best_oIoU = np.float(np.load((os.path.join(log_dir, 'best_performance.npy'))))
    else:
        resume_epoch = -999
    log_string('Training begin, the previous best_performance is {}'.format(best_oIoU))

    iou, overallIoU = evaluate(model, data_loader_test, bert_model)
    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model)

        log_string('Average object IoU {}'.format(iou))
        log_string('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            log_string('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU
            log_string('The best_performance is {}'.format(best_oIoU))
            np.save((os.path.join(log_dir, 'best_performance.npy')), best_oIoU.cpu().numpy())

    # summarize
    log_string('The final_best_performance is {}'.format(best_oIoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log_string('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    setup_seed(3407)
    log_dir = os.path.join(args.rootpath, args.model_id)
    LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'a')
    resume_flag = os.path.exists(os.path.join(log_dir, 'model_best_') + args.model_id + '.pth')
    # pdb.set_trace()
    if resume_flag:
        cp_path = os.path.join(log_dir, 'model_best_') + args.model_id + '.pth'
    # pdb.set_trace()
    # copy_dirs(log_dir)
    # shutil.copytree('./lib/', os.path.join(work_dir, 'lib'))
    # os.system('cp %s %s' % ('./lib/', log_dir))
    # LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'w')
    log_string('Image size: {}'.format(str(args.img_size)))
    main(args)
