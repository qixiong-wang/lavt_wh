import os
import torch
import numpy as np
import cv2
# dir1 = 'input_ms_output'
# dir2 = 'model_ms_output'
dir1 = 'input_ms_mask'
dir2 = 'model_ms_mask'

filename_list = sorted(os.listdir(dir1))
for filename in filename_list:
    output1= torch.load(os.path.join(dir1,filename))
    output2= torch.load(os.path.join(dir2,filename))
#     output_mask1 = output1.argmax(1)
#     output_mask2 = output2.argmax(1)
    # print(torch.sum(output1!=output2))
    # print(torch.sum(output_mask1!=output_mask2))
        
    print(np.sum(output1!=output2))
    if np.sum(output1!=output2)>10000:
        mask_save = np.concatenate((output1,output2),axis=2)
        mask_save = np.repeat(mask_save,3,axis=0)*255
        mask_save = mask_save.astype(np.uint8)
        mask_save = mask_save.transpose(1,2,0)
        cv2.imwrite(filename.replace('pkl','png'), mask_save)