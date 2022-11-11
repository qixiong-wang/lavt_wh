import numpy as np
from PIL import Image
import pdb
import glob

input_path = '/mnt/lustre/share_data/huyutao/coco2014/train2014/'
test_gt_path = '/mnt/lustre/huyutao.vendor/record/lavit/pred1110/vis_target_data/'
test_pred_path = '/mnt/lustre/huyutao.vendor/record/lavit/pred1110/vis_output_mask_data/'
save_path = '/mnt/lustre/huyutao.vendor/record/lavit/pred1110/pinjie'

ff = glob.glob(test_gt_path + '*.png')
length = len(ff)
print(length)
for i in range(length):
    name = ff[i]
    text = name.split('_')[-1].split('.')[0]
    img_path = input_path + 'COCO_train2014_' + name.split('/')[-1].split('_')[2] + '.jpg'
    name1 = name.split('/')[-1]
    gt_path = test_gt_path + name1
    pd_path = test_pred_path + name1

    img = Image.open(img_path).resize((480, 480))
    img1 = np.array(img)
    if len(img1.shape) < 3:
        img1 = img1[:, :, np.newaxis]
        img1 = img1.repeat([3], axis=2)

    gt = Image.open(gt_path)
    gt1 = np.array(gt)
    gt1 = gt1[:, :, np.newaxis]
    gt2 = gt1.repeat([3], axis=2)

    pred = Image.open(pd_path)
    pd1 = np.array(pred)
    pd1 = pd1[:, :, np.newaxis]
    pd2 = pd1.repeat([3], axis=2)
    save_name = '/COCO_train2014_' + name.split('/')[-1].split('_')[2] + text + '.jpg'
    final_img = np.concatenate((img1, gt2, pd2), axis=1)
    final_img = Image.fromarray(final_img)
    final_img.save(save_path + save_name)
    # pdb.set_trace()
    print(i)