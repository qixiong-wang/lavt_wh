import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


# Copyright (c) OpenMMLab. All rights reserved.
from tkinter.tix import Tree
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import ModuleList
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)



class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)
        self.conv_emb = nn.Conv2d(hidden_size, 768, 1)

        self.num_classes = 2

    def forward(self, x_c4, x_c3, x_c2, x_c1):

        # fuse Y4 and Y3

        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        embedding = self.conv_emb(x)
        embedding = F.adaptive_avg_pool2d(embedding,output_size=1)
        output = self.conv1_1(x)
        # output = x

        # b, c, h, w = output.shape
        # output = output.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        # cls_seg_feat = self.cls_emb.expand(output.size(0), -1, -1)
        # cls_seg_feat = self.transformer_cross_attention_layers(cls_seg_feat,output)
        # cls_seg_feat = self.transformer_self_attention_layers(cls_seg_feat)
        # cls_seg_feat = self.transformer_ffn_layers(cls_seg_feat)

        # output = self.patch_proj(output)
        # cls_seg_feat = self.classes_proj(cls_seg_feat)

        # output = F.normalize(output, dim=2, p=2)
        # cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)
        
        # output = output @ cls_seg_feat.transpose(1, 2)
        # output = self.mask_norm(output)
        # output = output.permute(0, 2, 1).contiguous().view(b,-1, h, w)
        return output,embedding
