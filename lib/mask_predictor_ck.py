import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
from torch.autograd import Variable
from copy import deepcopy
import pdb
from collections import OrderedDict


class ckattention(nn.Module):
    def __init__(self, la_dims, ke_dims, vi_dims):
        super(ckattention, self).__init__()

        self.f_lan = nn.Sequential(
            nn.Conv2d(la_dims, ke_dims, 3, padding=1, bias=False),
            nn.BatchNorm2d(ke_dims),
            nn.ReLU(),
        )
        self.f_vis = nn.Sequential(
            nn.Conv2d(vi_dims, ke_dims, 3, padding=1, bias=False),
            nn.BatchNorm2d(ke_dims),
            nn.ReLU(),
        )
        self.f_ker1 = nn.Sequential(
            nn.Conv2d(ke_dims, ke_dims, 1, bias=False),
            nn.LayerNorm([ke_dims, 1, 1]),
        )
        self.f_ker2 = nn.Sequential(
            nn.Conv2d(ke_dims, ke_dims, 1, bias=False),
            nn.LayerNorm([ke_dims, 1, 1]),
        )


    def forward(self, lan, kernel, vis):
        x_l = lan # BCWH
        x_k = kernel #KC
        x_v = vis #BCWH

        x_l = self.f_lan(x_l)
        x_v = self.f_vis(x_v)

        x_l1 = x_l.view(x_l.shape[0], x_l.shape[1], -1)
        x_v1 = x_v.view(x_v.shape[0], x_v.shape[1], -1)
        # pdb.set_trace()

        x1 = torch.einsum('bcn, kc->bkn', x_l1, x_k) #BKN
        x1 = F.softmax(x1, dim=-2)
        # x_k2 = torch.einsum('bkn, bcn->kc', x1, x_l1) #K*C
        x2 = torch.einsum('bkn, bcn->kc', x1, x_l1).view(x_k.shape[0], x_k.shape[1], 1, 1) #K*C*1*1
        x_k2 = self.f_ker1(x2).view(x2.shape[0], x2.shape[1])

        x3 = torch.einsum('bcn, kc->bkn', x_v1, x_k2) #BKN
        x3 = F.softmax(x3, dim=-2)
        # x4 = torch.einsum('bkn, bcn->kc', x3, x_v1) #K*C
        x4 = torch.einsum('bkn, bcn->kc', x3, x_v1).view(x_k.shape[0], x_k.shape[1], 1, 1) #K*C*1*1
        x_k4 = self.f_ker2(x4).view(x4.shape[0], x4.shape[1])

        return x_k4



class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.init_ker = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 1, bias=False),
            # nn.LayerNorm([hidden_size, 1, 1]),
        )

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.ck_learn1 = ckattention(c2_size, hidden_size, hidden_size)

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.ck_learn2 = ckattention(c1_size, hidden_size, hidden_size)

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)
        # self.convfinal = nn.Conv2d(4, 2, 1)
        # pdb.set_trace()
        # self.kernel = r
        # self.kernel = torch.randn((2, 512), out=None, dtype=None, layout=torch.strided, device="cuda", requires_grad=True)
        self.kernel = nn.Parameter(torch.randn(2, 512))
        trunc_normal_(self.kernel, std=.02)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # pdb.set_trace()
        # fuse Y4 and Y3
        # ker = self.init_ker(self.kernel.view(self.kernel.shape[0], self.kernel.shape[1], 1, 1))
        # ker = ker.view(self.kernel.shape[0], self.kernel.shape[1])
        # pdb.set_trace()
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)

        # fuse top-down features and Y2 features and pre1
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        ker = self.ck_learn1(x_c2, self.kernel, x)
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
        ker = self.ck_learn2(x_c1, ker, x)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        # x1 = self.conv1_1(x)
        # x = torch.einsum('bcwh, kc->bkwh', x, ker)
        # x = torch.cat([x1, x], dim=1)
        # x = self.convfinal(x)

        x = x.permute(0, 2, 3, 1) @ ker.permute(1, 0)
        x = x.permute(0, 3, 1, 2)


        return x
