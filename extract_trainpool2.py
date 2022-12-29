import csv
import pdb
from shutil import copyfile
import os
import numpy as np

refcoco = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcoco.csv', 'r')
refcoco1 = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcoco+.csv', 'r')
refcocog = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcocog.csv', 'r')

ff = open('/data/huyutao/new_coco/dataset_split_true_round2/refseg_train_pools.csv', 'w')
ff1 = csv.writer(ff)
ff1.writerow(['index', 'source', 'filename'])

ffwo = open('/data/huyutao/new_coco/dataset_split_true_round2/refseg_train_pools_wotest.csv', 'w')
ffwo1 = csv.writer(ffwo)
ffwo1.writerow(['index', 'source','filename'])

index = 0
train_list_all = []
train_list_wotest = []
train_list_num = 0
train_list_wo_num = 0

test_pool = np.load('/data/huyutao/new_coco/dataset_split_true_round2/refseg_valtest_pools.npy')

for line in refcoco:
    if index == 0:
        flag = line.split(',')[1]
        index = index + 1
    else:
        name_now = line.split(',')[1]
        if name_now != flag:
            flag = name_now
            aa = name_now.split('_')
            bb = '_'
            name_now = bb.join(aa[:-1]) + '.jpg'
            if name_now not in train_list_all:
                train_list_num = train_list_num + 1
                train_list_all.append(name_now)
                ff1.writerow([train_list_num, 'refcoco', name_now])
                if name_now not in test_pool:
                    train_list_wo_num = train_list_wo_num + 1
                    train_list_wotest.append(name_now)
                    ffwo1.writerow([train_list_wo_num, 'refcoco', name_now])

index = 0
print('refcoco done')
refcoco_num = train_list_num
refcoco_num_wo = train_list_wo_num

for line in refcoco1:
    if index == 0:
        flag = line.split(',')[1]
        index = index + 1
    else:
        name_now = line.split(',')[1]
        if name_now != flag:
            flag = name_now
            aa = name_now.split('_')
            bb = '_'
            name_now = bb.join(aa[:-1]) + '.jpg'
            if name_now not in train_list_all:
                train_list_num = train_list_num + 1
                train_list_all.append(name_now)
                ff1.writerow([train_list_num, 'refcoco+', name_now])
                if name_now not in test_pool:
                    train_list_wo_num = train_list_wo_num + 1
                    train_list_wotest.append(name_now)
                    ffwo1.writerow([train_list_wo_num, 'refcoco+', name_now])

index = 0
print('refcoco1 done')
refcoco1_num = train_list_num - refcoco_num
refcoco1_num_wo = train_list_wo_num - refcoco_num_wo

for line in refcocog:
    if index == 0:
        flag = line.split(',')[1]
        index = index + 1
    else:
        name_now = line.split(',')[1]
        if name_now != flag:
            flag = name_now
            aa = name_now.split('_')
            bb = '_'
            name_now = bb.join(aa[:-1]) + '.jpg'
            if name_now not in train_list_all:
                train_list_num = train_list_num + 1
                train_list_all.append(name_now)
                ff1.writerow([train_list_num, 'refcocog', name_now])
                if name_now not in test_pool:
                    train_list_wo_num = train_list_wo_num + 1
                    train_list_wotest.append(name_now)
                    ffwo1.writerow([train_list_wo_num, 'refcocog', name_now])

index = 0
print('refcocog done')
refcocog_num = train_list_num - refcoco_num - refcoco1_num
refcocog_num_wo = train_list_wo_num - refcoco_num_wo - refcoco1_num_wo

pdb.set_trace()
print('total train number is' + str(train_list_num) + 'total train number wo test is' + str(train_list_wo_num))
print('refcoco used train number is' + str(refcoco_num) + 'refcoco used train number wo test is' + str(refcoco_num_wo))
print('refcoco+ used train number is' + str(refcoco1_num) + 'refcoco+ used train number wo test is' + str(refcoco1_num_wo))
print('refcocog used train number is' + str(refcocog_num) + 'refcocog used train number wo test is' + str(refcocog_num_wo))
pdb.set_trace()
np.save('/data/huyutao/new_coco/dataset_split_true_round2/refseg_train_pools.npy', train_list_all)
np.save('/data/huyutao/new_coco/dataset_split_true_round2/refseg_train_pools_wo_test.npy', train_list_wotest)
