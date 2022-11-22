import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import os, torch, pdb
import math


def visulize_attention(mask, length, filename, sentence):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    #
    # mean=[0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    #
    # mean = np.array(mean)
    # mean = np.expand_dims(mean, 1)
    # mean = np.expand_dims(mean, 2)
    # std = np.array(std)
    # std = np.expand_dims(std, 1)
    # std = np.expand_dims(std, 2)
    #
    # # prepare the image
    # img = img.squeeze(0)
    # img = np.array(img.cpu())
    # img = (img * std + mean) * 255
    # img = img.transpose(1, 2, 0)
    # img = np.uint8(img)
    # img1 = Image.fromarray(img)
    # img1.save('./img.jpg')

    wh = mask.shape[-2]
    mask = mask.squeeze(0)
    mask = mask.squeeze(0)
    m1 = mask[:, :length]
    m1 = torch.sum(m1, 1)/length
    m2 = m1.view(int(math.sqrt(m1.shape[0])), int(math.sqrt(m1.shape[0])))
    m2 = np.array(m2.cpu())
    m3 = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2))
    m3 = np.uint8(m3 * 255)
    m4 = cv2.applyColorMap(m3, cv2.COLORMAP_JET)
    m4 = m4[:, :, [2, 1, 0]]
    m5 = Image.fromarray(m4)
    # m5.save('./img0.jpg')

    save_output_dir = '/mnt/petrelfs/huyutao/record/lavit_hu1/vis_attention_matrix/'
    if not os.path.exists(save_output_dir):
        # import pdb
        # pdb.set_trace()
        os.mkdir(save_output_dir)
    # m5.save(save_output_dir + str(1).zfill(6) + '.jpg')
    m5.save(save_output_dir + filename.replace('.jpg', '_') + sentence + str(wh) + '.jpg')
    # pdb.set_trace()

def visulize_attention_lan(mask, length, filename, sentence):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    #
    # mean=[0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    #
    # mean = np.array(mean)
    # mean = np.expand_dims(mean, 1)
    # mean = np.expand_dims(mean, 2)
    # std = np.array(std)
    # std = np.expand_dims(std, 1)
    # std = np.expand_dims(std, 2)
    #
    # # prepare the image
    # img = img.squeeze(0)
    # img = np.array(img.cpu())
    # img = (img * std + mean) * 255
    # img = img.transpose(1, 2, 0)
    # img = np.uint8(img)
    # img1 = Image.fromarray(img)
    # img1.save('./img.jpg')

    nl = mask.shape[-1]
    mask = mask.squeeze(0)
    mask = mask.squeeze(0)
    m1 = mask[:, :length]
    m1 = torch.sum(m1, 0)/length
    m2 = m1.view(m1.shape[0], 1)
    m2 = np.array(m2.cpu())
    m3 = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2))
    m3 = np.uint8(m3 * 255)
    m4 = cv2.applyColorMap(m3, cv2.COLORMAP_JET)
    m4 = m4[:, :, [2, 1, 0]]
    m5 = Image.fromarray(m4)
    # m5.save('./img0.jpg')

    save_output_dir = '/mnt/petrelfs/huyutao/record/lavit_hu1/vis_attention_matrix/'
    if not os.path.exists(save_output_dir):
        # import pdb
        # pdb.set_trace()
        os.mkdir(save_output_dir)
    # m5.save(save_output_dir + str(1).zfill(6) + '.jpg')
    m5.save(save_output_dir + filename.replace('.jpg', '_') + 'lan_' + sentence + str(nl) + '.jpg')
    # pdb.set_trace()


def visulize_attention1(idx, img, pred):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """

    mean=[0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = np.array(mean)
    mean = np.expand_dims(mean, 1)
    mean = np.expand_dims(mean, 2)
    std = np.array(std)
    std = np.expand_dims(std, 1)
    std = np.expand_dims(std, 2)

    # prepare the image
    img = img.squeeze(0)
    img = np.array(img.cpu())
    img = (img * std + mean) * 255
    img = img.transpose(1, 2, 0)
    img = np.uint8(img)
    # img1 = Image.fromarray(img)
    # img1.save('./img.jpg')

    pred = pred.squeeze(0)
    pp1 = torch.argmax(pred, dim=0)
    m2 = pp1
    m2 = np.array(m2.cpu())
    m3 = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2))
    m3 = np.uint8(m3 * 255)
    m4 = m3[:, :, np.newaxis]
    m4 = m4.repeat([3], axis=2)
    f0 = np.concatenate((img, m4), axis=1)
    f1 = Image.fromarray(f0)

    save_output_dir = '/mnt/petrelfs/huyutao/record/lavit_hu1/vis_h18_1115/'
    if not os.path.exists(save_output_dir):
        # import pdb
        # pdb.set_trace()
        os.mkdir(save_output_dir)
    f1.save(save_output_dir + str(idx).zfill(6) + '.jpg')


    # m5 = Image.fromarray(m3)
    # m5.save('./pred.jpg')
    # pdb.set_trace()

    # mask = mask.squeeze(0)
    # mask = mask.squeeze(0)
    # m1 = mask[:, :length]
    # m1 = torch.sum(m1, 1)/length
    # m2 = m1.view(int(math.sqrt(m1.shape[0])), int(math.sqrt(m1.shape[0])))
    # m2 = np.array(m2.cpu())
    # m3 = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2))
    # m3 = np.uint8(m3 * 255)
    # m4 = cv2.applyColorMap(m3, cv2.COLORMAP_JET)
    # m4 = m4[:, :, [2, 1, 0]]
    # m5 = Image.fromarray(m4)
    # m5.save('./img0.jpg')
    # pdb.set_trace()

