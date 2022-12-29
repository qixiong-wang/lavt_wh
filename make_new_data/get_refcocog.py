import csv, pdb, os, shutil
import numpy as np
import json



refcocog = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcocog_val.csv', 'r')
# refcocog = open('../ori_refcoco+_testA.csv', 'r')

index = 0
refcocog_train_pools = []
save_num = 0

for line in refcocog:
    if index == 0:
        flag = line.split(',')[1]
        index = index + 1
    else:
        name_now = line.split(',')[1]
        aa = name_now.split('_')
        bb = '_'
        name_now = bb.join(aa[:-1]) + '.jpg'
        if name_now not in refcocog_train_pools:
            refcocog_train_pools.append(name_now)
print(len(refcocog_train_pools))
pdb.set_trace()