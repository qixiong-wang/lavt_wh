import torch
from torch import nn
from torch.nn import functional as F
import pdb
from collections import OrderedDict
from .lan_decoder import simple_lan_transformer


class CycleDecode(nn.Module):
    def __init__(self, in_dims):
        super(CycleDecode, self).__init__()

        inter_dims = in_dims

        self.conv1 = nn.Conv2d(in_dims, inter_dims, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_dims)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(inter_dims, inter_dims, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_dims)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(inter_dims, 2, 1)

    def forward(self, input):
        x = input

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        return x

class CrossLayerFuse(nn.Module):
    def __init__(self, in_dims1, in_dims2, out_dims):
        super(CrossLayerFuse, self).__init__()

        # inter_dims = in_dims
        self.linear = nn.Linear(in_dims1 + in_dims2, out_dims)
        self.adpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, defea, x):
        x_pre = defea

        x = self.adpool(x).view(x.shape[0], x.shape[1])
        x1 = torch.cat([x_pre, x], dim=1)
        x1 = self.linear(x1)

        return x1

class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims//factor
        lan_size = 768
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.adpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        # self.cydecode1 = CycleDecode(hidden_size)

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()
        self.crossfuse1 = CrossLayerFuse(hidden_size, hidden_size, lan_size)

        # self.cydecode2 = CycleDecode(hidden_size)


        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)
        self.lan_func = simple_lan_transformer(hidden_size, lan_size=768)
        self.crossfuse2 = CrossLayerFuse(lan_size, hidden_size, lan_size)


    def forward(self, lan, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x) # [B, 512, 30, 30]
        defea = self.adpool(x).view(x.shape[0], x.shape[1])
        # print(444444444444)
        # pdb.set_trace()

        # pre1 = self.cydecode1(x) ## pre1 [B, 512, 30, 30]

        # fuse top-down features and Y2 features and pre1
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        # if pre1.size(-2) < x_c2.size(-2) or pre1.size(-1) < x_c2.size(-1):
        #     pre1 = F.interpolate(input=pre1, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x) # [B, 512, 60, 60]

        new_lan = self.lan_func(x, lan)
        defea = self.crossfuse1(defea, x)


        # pre2 = self.cydecode2(x) ## pre1 [B, 512, 60, 60]

        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        # if pre2.size(-2) < x_c1.size(-2) or pre2.size(-1) < x_c1.size(-1):
        #     pre2 = F.interpolate(input=pre2, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x) # [B, 512, 120, 120]
        defea = self.crossfuse2(defea, x)


        return defea, new_lan, self.conv1_1(x)
