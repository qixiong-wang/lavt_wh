import csv, pdb, os, shutil
import numpy as np
import json


def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)

train_wotest_multar_claname_pools = np.load('/data/huyutao/new_coco/train_wotest_multar_claname_filepools.npy')
train_wotest_multar_claname_idpools = np.load('/data/huyutao/new_coco/train_wotest_multar_claname_idpools.npy')
train_wotest_pools = np.load('/data/huyutao/new_coco/refseg_train_pools_wo_test.npy')
cap_annos = json.load(open('/data/huyutao/coco2014/annotations/captions_train2014.json', 'r'))

inputfile = '/data/huyutao/coco2014/train2014'
outputfile = '/data/huyutao/new_coco/train_wotest_multar_cap'

mkdir_os(outputfile)

if len(train_wotest_multar_claname_pools) != len(train_wotest_multar_claname_idpools):
    print('error!!!')
else:
    print('OKK!!!!')

save_num = 0
used_filenames = []

length = len(cap_annos['annotations'])
for i in range(length):
    id_now = cap_annos['annotations'][i]['image_id']
    filename_now = 'COCO_train2014_' + str(id_now).zfill(12) + '.jpg'
    # pdb.set_trace()
    # print(i)
    if filename_now in train_wotest_multar_claname_pools:
        if filename_now not in used_filenames:
            # print(11111111)
            # filename_now = train_wotest_multar_claname_pools[i]
            cap_now = cap_annos['annotations'][i]['caption']
            # pdb.set_trace()
            new_filename = filename_now.split('.')[0] + '_' + cap_now  + '_' + '.jpg'
            source_path = os.path.join(inputfile, filename_now)
            target_path = os.path.join(outputfile, new_filename)
            # pdb.set_trace()
            try:
                shutil.copyfile(source_path, target_path)
                save_num = save_num + 1
                used_filenames.append(filename_now)
            except:
                continue

        # pdb.set_trace()
print('total number is' + str(save_num))


# train_wotest_multar_claname_filepools = []
# train_wotest_multar_claname_idpools = []
# index = 0

# ff = open('/data/huyutao/new_coco/train_wotest_multar_cla_name1.csv', 'r')
# for line in ff:
#     if index == 0:
#         index = index + 1
#     else:
#         name_now = line.split(',')[1]
#         id_now = line.split(',')[3]
#         train_wotest_multar_claname_filepools.append(name_now)
#         train_wotest_multar_claname_idpools.append(id_now[:-1])

# np.save('/data/huyutao/new_coco/train_wotest_multar_claname_filepools.npy', train_wotest_multar_claname_filepools)
# np.save('/data/huyutao/new_coco/train_wotest_multar_claname_idpools.npy', train_wotest_multar_claname_idpools)






