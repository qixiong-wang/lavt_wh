import csv
import pdb
from shutil import copyfile
import os


refcoco_val = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcoco_val.csv', 'r')
refcoco_testA = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcoco_testA.csv', 'r')
refcoco_testB = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcoco_testB.csv', 'r')
refcoco1_val = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcoco+_val.csv', 'r')
refcoco1_testA = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcoco+_testA.csv', 'r')
refcoco1_testB = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcoco+_testB.csv', 'r')
refcocog_val = open('/data/huyutao/new_coco/dataset_split_true_round2/ori_refcocog_val.csv', 'r')

refcoco_all = [refcoco_val, refcoco_testA, refcoco_testB]
refcoco1_all = [refcoco1_val, refcoco1_testA, refcoco1_testB]
refcocog_all = [refcocog_val]

refcoco_source = ['refcoco_val', 'refcoco_testA', 'refcoco_testB']
refcoco1_source = ['refcoco+_val', 'refcoco+_testA', 'refcoco+_testB']
refcocog_source = ['refcocog_val']

ff = open('/data/huyutao/new_coco/dataset_split_true_round2/refseg_valtest_pools.csv', 'w')
ff1 = csv.writer(ff)
ff1.writerow(['index', 'source', 'filename'])

index = 0
test_list = []
save_num = 0

for i in range(len(refcoco_all)):
    ref_csv_now = refcoco_all[i]
    index = 0
    for line in ref_csv_now:
        if index == 0:
            flag = line.split(',')[1]
            index = index + 1
        else:
            name_now = line.split(',')[1]
            aa = name_now.split('_')
            bb = '_'
            name_now = bb.join(aa[:-1]) + '.jpg'
            if name_now not in test_list:
                save_num = save_num + 1
                test_list.append(name_now)
                ff1.writerow([save_num, refcoco_source[i], name_now])
                # print(save_num)
index = 0
print('refcoco done')
refcoco_num = save_num

for i in range(len(refcoco1_all)):
    ref_csv_now = refcoco1_all[i]
    index = 0
    for line in ref_csv_now:
        if index == 0:
            flag = line.split(',')[1]
            index = index + 1
        else:
            name_now = line.split(',')[1]
            aa = name_now.split('_')
            bb = '_'
            name_now = bb.join(aa[:-1]) + '.jpg'
            if name_now not in test_list:
                save_num = save_num + 1
                test_list.append(name_now)
                ff1.writerow([save_num, refcoco1_source[i], name_now])
                # print(save_num)
index = 0
print('refcoco+ done')
refcoco1_num = save_num - refcoco_num

for i in range(len(refcocog_all)):
    ref_csv_now = refcocog_all[i]
    index = 0
    for line in ref_csv_now:
        if index == 0:
            flag = line.split(',')[1]
            index = index + 1
        else:
            name_now = line.split(',')[1]
            aa = name_now.split('_')
            bb = '_'
            name_now = bb.join(aa[:-1]) + '.jpg'
            if name_now not in test_list:
                save_num = save_num + 1
                test_list.append(name_now)
                ff1.writerow([save_num, refcocog_source[i], name_now])
                # print(save_num)
index = 0
refcocog_num = save_num - refcoco_num - refcoco1_num
pdb.set_trace()
print('refcocog done')
print('total test number is' + str(save_num))
print('refcoco used test number is' + str(refcoco_num))
print('refcoco1 used test number is' + str(refcoco1_num))
print('refcocog used test number is' + str(refcocog_num))

