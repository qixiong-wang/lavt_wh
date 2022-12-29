import csv
import pdb
from shutil import copyfile
import os

refcoco = open('/data/huyutao/new_coco/ori_refcoco_test.csv', 'r')
refcoco1 = open('/data/huyutao/new_coco/ori_refcoco+_test.csv', 'r')
refcocog = open('/data/huyutao/new_coco/ori_refcocog_test.csv', 'r')

ff = open('/data/huyutao/new_coco/refseg_test_pools.csv', 'w')
ff1 = csv.writer(ff)
ff1.writerow(['index', 'filename'])

index = 0
test_list = []
save_num = 0

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
            if name_now not in test_list:
                save_num = save_num + 1
                test_list.append(name_now)
                ff1.writerow([save_num, name_now])
                print(save_num)
index = 0
print('refcoco done')
refcoco_num = save_num
            
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
            if name_now not in test_list:
                save_num = save_num + 1
                test_list.append(name_now)
                ff1.writerow([save_num, name_now])
                print(save_num)
index = 0
print('refcoco+ done')
refcoco1_num = save_num - refcoco_num

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
            if name_now not in test_list:
                save_num = save_num + 1
                test_list.append(name_now)
                ff1.writerow([save_num, name_now])
                print(save_num)
index = 0
refcocog_num = save_num - refcoco_num - refcoco1_num
pdb.set_trace()
print('refcocog done')
print('total test number is' + str(save_num))
print('refcoco used test number is' + str(refcoco_num))
print('refcoco1 used test number is' + str(refcoco1_num))
print('refcocog used test number is' + str(refcocog_num))

