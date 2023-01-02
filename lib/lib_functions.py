import pdb
from cmath import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Nce_Contrast_Loss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    """
    def __init__(self, team=5, batch=8):
        super(Nce_Contrast_Loss, self).__init__()
        self.team = team
        self.adpool = nn.AdaptiveAvgPool2d((1, 1))
        self.batch = batch
        self.align_lan = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=1, stride=1),
        )


    def forward(self, vis_feature, lan_feature):
        """
        """
        #print(inputs.shape, targets.shape)
        vv = vis_feature # [B 768]
        la = lan_feature # [B, 768, 20]

        vv1 = F.normalize(vv, dim=1)
        la1 = self.align_lan(la)
        # la1 = la
        la1 = self.adpool(la1.unsqueeze(3)).view(la.shape[0], la.shape[1])
        la1 = F.normalize(la1, dim=1)
        # pdb.set_trace()

        # img_text_logits = F.softmax(torch.matmul(vv1, la1.permute(1, 0)) * 20, dim=1)
        # text_img_logits = F.softmax(torch.matmul(vv1, la1.permute(1, 0)) * 20, dim=0)
        img_text_logits = torch.matmul(vv1, la1.permute(1, 0)) * 20
        text_img_logits = img_text_logits.permute(1, 0)
        labels = torch.arange(0, self.batch).cuda()
        loss_a = nn.functional.cross_entropy(img_text_logits, labels)
        loss_b = nn.functional.cross_entropy(text_img_logits, labels)
        loss_con = loss_a + loss_b

        return loss_con
