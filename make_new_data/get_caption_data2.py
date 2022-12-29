import csv, pdb, os, shutil
import numpy as np
import json

root_path = '/data/huyutao/new_coco/dataset_split_true_round2/'


def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)

train_wotest_multar_claname_pools = np.load(root_path + 'train_wotest_filepools.npy')
train_wotest_multar_claname_idpools = np.load(root_path + 'train_wotest_idpools.npy')
testval_pools = np.load(root_path + 'testval_filepools.npy')
testval_idpools = np.load(root_path + 'testval_idpools.npy')
##### !!!select train 2014######
# cap_annos = json.load(open('/data/huyutao/coco2014/annotations/captions_train2014.json', 'r'))
# inputfile = '/data/huyutao/coco2014/train2014'
##### !!!select val 2014######
cap_annos = json.load(open('/data/huyutao/coco2014/annotations/captions_val2014.json', 'r'))
inputfile = '/data/huyutao/coco2014/val2014'

outputfile = root_path + 'val_cap_begin234567'
# outputfile = './'

mkdir_os(outputfile)

if len(train_wotest_multar_claname_pools) != len(train_wotest_multar_claname_idpools):
    print('error!!!')
else:
    print('OKK!!!!')
if len(testval_pools) != len(testval_idpools):
    print('error!!!')
else:
    print('OKK!!!!')

all_refcoco_pools = train_wotest_multar_claname_pools.tolist() + testval_pools.tolist()

save_num = 0
used_image = []
check_list = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven']

ff = open(root_path + 'val_cap_begin234567.csv', 'w')
ff1 = csv.writer(ff)
ff1.writerow(['index', 'filename', 'caption'])


length = len(cap_annos['annotations'])
for i in range(length):
    id_now = cap_annos['annotations'][i]['image_id']
    filename_now = 'COCO_val2014_' + str(id_now).zfill(12) + '.jpg'
    # print(i)
    # if filename_now not in all_refcoco_pools:
    if filename_now not in used_image:
    # if True:
        # print(11111111)
        cap_now = cap_annos['annotations'][i]['caption']
        # pdb.set_trace()
        first_word = cap_now[:cap_now.index(' ')]
        if first_word in check_list and 'are' not in cap_now and 'is' not in cap_now:
        # if first_word in check_list and 'are' in cap_now and 'is' in cap_now:
        # if cap_now[:3] == 'Two' or cap_now[:5] == 'Three' or cap_now[:4] == 'Four' or cap_now[:4] == 'Five' or cap_now[:7] == 'Several':
            # pdb.set_trace()
            new_filename = '3_' + str(save_num+1) + '_' + filename_now.split('.')[0] + '_' + cap_now  + '_' + '.jpg'
            source_path = os.path.join(inputfile, filename_now)
            target_path = os.path.join(outputfile, new_filename)
            # pdb.set_trace()
            try:
                shutil.copyfile(source_path, target_path)
                save_num = save_num + 1
                ff1.writerow([save_num, filename_now, cap_now])
                if filename_now not in used_image:
                    used_image.append(filename_now)
                print(save_num)
                # pdb.set_trace()
            except:
                continue
        if save_num >= 10000:
            break

    # pdb.set_trace()
print('total number is' + str(save_num))
print('total image number is' + str(len(used_image)))




# train_wotest_multar_claname_filepools = []
# train_wotest_multar_claname_idpools = []
# index = 0

# ff = open(root_path + '/train_wotest_multar_cla_name1.csv', 'r')
# for line in ff:
#     if index == 0:
#         index = index + 1
#     else:
#         name_now = line.split(',')[1]
#         id_now = line.split(',')[3]
#         train_wotest_multar_claname_filepools.append(name_now)
#         train_wotest_multar_claname_idpools.append(id_now[:-1])

# np.save(root_path + '/train_wotest_multar_claname_filepools.npy', train_wotest_multar_claname_filepools)
# np.save(root_path + '/train_wotest_multar_claname_idpools.npy', train_wotest_multar_claname_idpools)



# # train_wotest_filepools = []
# # train_wotest_idpools = []
# testval_filepools = []
# testval_idpools = []
# index = 0

# # ff = open(root_path + 'refseg_train_pools_wotest.csv', 'r')
# ff = open(root_path + 'refseg_valtest_pools.csv', 'r')
# for line in ff:
#     if index == 0:
#         index = index + 1
#     else:
#         name_now = line.split(',')[2][:-1]
#         id_now = int(line.split(',')[2].split('.')[0].split('_')[2])
#         # pdb.set_trace()
#         testval_filepools.append(name_now)
#         testval_idpools.append(id_now)

# np.save(root_path + '/testval_filepools.npy', testval_filepools)
# np.save(root_path + '/testval_idpools.npy', testval_idpools)
# pdb.set_trace()
