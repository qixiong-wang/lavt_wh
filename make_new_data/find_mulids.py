# -*- coding: utf-8 -*-
import os
import sys, getopt
from pycocotools.coco import COCO
import cv2, pdb, csv
import matplotlib.pyplot as plt
import numpy as np

def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(argv):
    inputfile = '/data/huyutao/coco2014/train2014'   # './data/coco/val2017/'
    jsonfile = '/data/huyutao/coco2014/annotations/instances_train2014.json'    # './data/coco/annotations/instances_val2017.json'
    outputfile = '/data/huyutao/new_coco/multi_targets_cla_name1/'  # './data/coco/vis/'

    try:
        opts, args = getopt.getopt(argv, "hi:j:o:", ["ifile=", "jfile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -j <jsonfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -j <jsonfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-j", "--jfile"):
            jsonfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    print('\n输入的文件为：', inputfile)
    print('\n输入的json为：', jsonfile)
    print('\n输出的文件为：', outputfile)

    train_wo_pool = np.load('/data/huyutao/new_coco/refseg_train_pools_wo_test.npy')

    mkdir_os(outputfile)

    coco_clas = np.load('/data/huyutao/new_coco/coco2014_cla_ids.npy', allow_pickle=True).item()

    ff = open('/data/huyutao/new_coco/train_wotest_multar_cla_name1.csv', 'w')
    ff1 = csv.writer(ff)
    ff1.writerow(['index', 'filename', 'category', 'img_id'])

    coco = COCO(jsonfile)
    catIds = coco.getCatIds(catNms=['wires'])  # catIds=1 表示人这一类
    # catIds = coco.getCatIds(catNms=coco_clas)  # catIds=1 表示人这一类
    # pdb.set_trace()
    imgIds = coco.getImgIds(catIds=catIds)  # 图片id，许多值
    print(imgIds)
    use_num = 0

    for i, imgId in enumerate(imgIds):
        print(i, "/", len(imgIds))
        img = coco.loadImgs(imgId)[0]
        cat_id_now = []
        cat_id_fre = {}
        cat_id_area = {}
        if img['file_name'] in train_wo_pool:

            cvImage = cv2.imread(os.path.join(inputfile, img['file_name']), -1)
            # pdb.set_trace()
            if len(cvImage.shape) < 3:
                m4 = cvImage[:, :, np.newaxis]
                m4 = m4.repeat([3], axis=2)
                cvImage = m4

##########################ori_vis_related###############################
            # cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
            # cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR)
##########################ori_vis_related###############################

##########################ori_vis_related###############################
            # plt.cla()
            # plt.axis('off')
            # plt.imshow(cvImage) 
##########################ori_vis_related###############################


            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)

##########################ori_vis_related###############################
            # coco.showAnns(anns)
##########################ori_vis_related###############################

            for j in range(len(anns)):
                id_now = anns[j]['category_id']
                area_now = anns[j]['area']
                cat_id_now.append(id_now)
                if id_now not in cat_id_fre:
                    cat_id_fre[id_now] = 1
                    cat_id_area[id_now] = area_now
                else:
                    cat_id_fre[id_now] = cat_id_fre[id_now] + 1
                    cat_id_area[id_now] = cat_id_area[id_now] + area_now
                
            # plt.savefig(os.path.join(outputfile, img['file_name'])) 
            maxarea_pos = np.argmax(list(cat_id_area.values()))
            maxarea_id = list(cat_id_fre.keys())[maxarea_pos]
            maxarea_fre = cat_id_fre[maxarea_id]
            if maxarea_fre > 1:
                use_num = use_num + 1
                ff1.writerow([use_num, img['file_name'], coco_clas[maxarea_id], img['id']])
                # pdb.set_trace()
                # target_id_pos = []
                # for j in range(len(anns)):
                #     if cat_id_now[j] == maxarea_id:
                #         target_id_pos.append(j)

                # mask = coco.annToMask(anns[target_id_pos[0]])
                # for j in range(1, len(target_id_pos)):
                #     mask += coco.annToMask(anns[target_id_pos[j]])

                # mask1 = mask[:, :, np.newaxis]
                # mask1 = mask1.repeat([3], axis=2)
                # mask2 = 255 * (mask1 != 0)
                # final = np.concatenate((cvImage, mask2), axis=1)
                # cat_name = coco_clas[maxarea_id]
                # cv2.imwrite(outputfile + '_'.join(img['file_name'].split('.')[:-1]) + '_' + cat_name +'.jpg', final)
                
                # pdb.set_trace() 
    print('total number is' + str(use_num))



if __name__ == "__main__":
    main(sys.argv[1:])
