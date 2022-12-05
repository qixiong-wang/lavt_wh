import csv
import pdb
from shutil import copyfile
import os

ff = open('/mnt/cache/huyutao.vendor/code/new_data/try.csv', 'r')
rootpath = '/mnt/lustre/share_data/huyutao/coco2014/train2014'
savedir = '/mnt/lustre/huyutao.vendor/dataset/new_coco/refcoco_or_train'
index = 0
save_num = 1
for line in ff:
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
            # pdb.set_trace()
            source = os.path.join(rootpath, name_now)
            target = os.path.join(savedir, name_now)
            copyfile(source, target)
            print(save_num)
            save_num = save_num + 1
        # index = index + 1
        # pdb.set_trace()