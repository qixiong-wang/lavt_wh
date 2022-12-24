import torch
from torch import nn
from torch.nn import functional as F
import pdb
from collections import OrderedDict


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


class Conv_Block(nn.Module):
    def __init__(self, input_size, inter_size, output_size):
        super(Conv_Block, self).__init__()

        self.conv1 = nn.Conv2d(input_size, inter_size, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(inter_size, output_size, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Conv_Block_Small(nn.Module):
    def __init__(self, input_size, inter_size, output_size):
        super(Conv_Block_Small, self).__init__()

        self.conv1 = nn.Conv2d(input_size, inter_size, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_size)
        self.relu1 = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x

class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims // factor
        lan_size = 768
        c4_size = c4_dims
        c3_size = c4_dims // (factor ** 1)
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)

        self.adpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        # self.bn1_4 = nn.BatchNorm2d(hidden_size)
        # self.relu1_4 = nn.ReLU()
        # self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        # self.bn2_4 = nn.BatchNorm2d(hidden_size)
        # self.relu2_4 = nn.ReLU()
        self.conv_block1 = Conv_Block(c4_size + c3_size, hidden_size, hidden_size)


        # self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        # self.bn1_3 = nn.BatchNorm2d(hidden_size)
        # self.relu1_3 = nn.ReLU()
        # self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        # self.bn2_3 = nn.BatchNorm2d(hidden_size)
        # self.relu2_3 = nn.ReLU()
        self.conv_block2 = Conv_Block(hidden_size + c2_size, hidden_size, hidden_size)

        self.crossfuse1 = CrossLayerFuse(hidden_size, hidden_size, lan_size)


        # self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        # self.bn1_2 = nn.BatchNorm2d(hidden_size)
        # self.relu1_2 = nn.ReLU()
        # self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        # self.bn2_2 = nn.BatchNorm2d(hidden_size)
        # self.relu2_2 = nn.ReLU()
        self.conv_block3 = Conv_Block(hidden_size + c1_size, hidden_size, hidden_size)
        self.conv_block4 = Conv_Block_Small(hidden_size, hidden_size // 2, hidden_size // 2)


        self.conv_f1 = nn.Conv2d(hidden_size // 2, 2, 3, padding=1, bias=False)

        self.crossfuse2 = CrossLayerFuse(lan_size, hidden_size, lan_size)


    def forward(self, x_c4, x_c3, x_c2, x_c1, input_shape):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        # x = self.conv1_4(x)
        # x = self.bn1_4(x)
        # x = self.relu1_4(x)
        # x = self.conv2_4(x)
        # x = self.bn2_4(x)
        # x = self.relu2_4(x)  # [B, 512, 30, 30]
        x = self.conv_block1(x) # [B, 512, 30, 30] if input is 480
        defea = self.adpool(x).view(x.shape[0], x.shape[1])


        # fuse top-down features and Y2 features and pre1
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        # if pre1.size(-2) < x_c2.size(-2) or pre1.size(-1) < x_c2.size(-1):
        #     pre1 = F.interpolate(input=pre1, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        # x = self.conv1_3(x)
        # x = self.bn1_3(x)
        # x = self.relu1_3(x)
        # x = self.conv2_3(x)
        # x = self.bn2_3(x)
        # x = self.relu2_3(x)  # [B, 512, 60, 60]
        x = self.conv_block2(x) # [B, 512, 60, 60] if input is 480

        defea = self.crossfuse1(defea, x)



        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        # if pre2.size(-2) < x_c1.size(-2) or pre2.size(-1) < x_c1.size(-1):
        #     pre2 = F.interpolate(input=pre2, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        # x = self.conv1_2(x)
        # x = self.bn1_2(x)
        # x = self.relu1_2(x)
        # x = self.conv2_2(x)
        # x = self.bn2_2(x)
        # x = self.relu2_2(x)  # [B, 512, 120, 120]
        x = self.conv_block3(x) # [B, 512, 120, 120] if input is 480


        defea = self.crossfuse2(defea, x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_block4(x) # [B, 256, 240, 240] if input is 480
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        x = self.conv_f1(x)


        return defea, x
