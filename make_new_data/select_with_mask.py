# -*- coding: utf-8 -*-
import numpy as np
import os, glob, pdb, cv2, shutil
import json
import sys, getopt
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)

input_path = '/data/huyutao/new_coco/dataset_split_true_round2/select/val_cap_begin234567_select'
output_path = '/data/huyutao/new_coco/dataset_split_true_round2/select/val_cap_begin234567_select_withmask'
file_names = glob.glob(os.path.join(input_path, '*.jpg'))
img_pool = []

jsonfile = '/data/huyutao/coco2014/annotations/instances_val2014.json'    # './data/coco/annotations/instances_val2017.json'
coco = COCO(jsonfile)
save_num = 0

mkdir_os(output_path)

for i in range(len(file_names)):
    file_now = file_names[i]
    img_now = '_'.join(file_now.split('/')[7].split('_')[2:5]) + '.jpg'
    img_id = int(img_now.split('.')[0].split('_')[2])
    annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
    if len(annIds) > 0:
        save_num = save_num + 1
        source_path = file_now
        target_name = file_now.split('/')[7:]
        if len(target_name) > 1:
            target_name1 = '/'.join(target_name[:])
        else:
            target_name1 = target_name[0]

        target_path = os.path.join(output_path, target_name1)
        shutil.copyfile(source_path, target_path)

    # pdb.set_trace()
print('before find mask is: ' + str(len(file_names)))
print('after find mask is: ' + str(save_num))







# inputfile = '/data/huyutao/coco2014/train2014'   # './data/coco/val2017/'
# jsonfile = '/data/huyutao/coco2014/annotations/instances_train2014.json'    # './data/coco/annotations/instances_val2017.json'
# outputfile = './'  # './data/coco/vis/'

# try:
#     opts, args = getopt.getopt(sys.argv[1:], "hi:j:o:", ["ifile=", "jfile=", "ofile="])
# except getopt.GetoptError:
#     print('test.py -i <inputfile> -j <jsonfile> -o <outputfile>')
#     sys.exit(2)
# for opt, arg in opts:
#     if opt == '-h':
#         print('test.py -i <inputfile> -j <jsonfile> -o <outputfile>')
#         sys.exit()
#     elif opt in ("-i", "--ifile"):
#         inputfile = arg
#     elif opt in ("-j", "--jfile"):
#         jsonfile = arg
#     elif opt in ("-o", "--ofile"):
#         outputfile = arg

# print('\n输入的文件为：', inputfile)
# print('\n输入的json为：', jsonfile)
# print('\n输出的文件为：', outputfile)

# mkdir_os(outputfile)

# coco = COCO(jsonfile)
# catIds = coco.getCatIds(catNms=['wires'])  # catIds=1 表示人这一类
# imgIds = coco.getImgIds(catIds=catIds)  # 图片id，许多值
# print(imgIds)

# masked_select_img = []
# for i, imgId in enumerate(imgIds):
#     img = coco.loadImgs(imgId)[0]
#     if img['file_name'] in img_pool:
#         # print('gogogogo!!!')
#         # print(i, "/", len(imgIds))

#         cvImage = cv2.imread(os.path.join(inputfile, img['file_name']), -1)
#         # img_file_name = '_'.join(img['file_name'].split('_')[:3])+'.jpg'
#         # cvImage = cv2.imread(os.path.join(inputfile, img_file_name), -1)
#         cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
#         cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR)

#         plt.cla()
#         plt.axis('off')
#         plt.imshow(cvImage) 

#         annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#         if len(annIds) > 0:
#             masked_select_img.append(img['file_name'])

# pdb.set_trace()
        # anns = coco.loadAnns(annIds)
        # coco.showAnns(anns)
        # save_path = outputfile + 'try2_' + img['file_name']
        # plt.savefig(save_path)








# json_path = '/data/huyutao/coco2014/annotations/instances_train2014.json'
# json_labels = json.load(open(json_path, "r"))
# pdb.set_trace()