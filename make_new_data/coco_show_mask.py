# -*- coding: utf-8 -*-
import os
import sys, getopt
from pycocotools.coco import COCO
import cv2, pdb
import matplotlib.pyplot as plt


def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(argv):
    inputfile = '/data/huyutao/coco2014/train2014'   # './data/coco/val2017/'
    jsonfile = '/data/huyutao/coco2014/annotations/instances_train2014.json'    # './data/coco/annotations/instances_val2017.json'
    outputfile = './'  # './data/coco/vis/'

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

    mkdir_os(outputfile)

    coco = COCO(jsonfile)
    # catIds = coco.getCatIds(catNms=['wires'])  # catIds=1 表示人这一类
    catIds = coco.getCatIds(catNms=['cow'])  # catIds=1 表示人这一类
    imgIds = coco.getImgIds(catIds=catIds)  # 图片id，许多值
    print(imgIds)

    for i, imgId in enumerate(imgIds):
        img = coco.loadImgs(imgId)[0]
        if img['file_name'] == 'COCO_train2014_000000007601.jpg':
            print('gogogogo!!!')
            # print(i, "/", len(imgIds))

            cvImage = cv2.imread(os.path.join(inputfile, img['file_name']), -1)
            cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
            cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR)

            plt.cla()
            plt.axis('off')
            plt.imshow(cvImage) 

            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            pdb.set_trace()
            coco.showAnns(anns)
            save_path = outputfile + 'try_' + img['file_name']
            plt.savefig(save_path)



if __name__ == "__main__":
    main(sys.argv[1:])
