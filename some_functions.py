import pdb
from cmath import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Nce_contrast_loss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    """
    def __init__(self, team=5, batch=32):
        super(Nce_contrast_loss, self).__init__()
        self.team = team
        self.adpool = nn.AdaptiveAvgPool2d((1, 1))
        self.batch = batch


    def forward(self, vis_feature, lan_feature):
        """
        """
        #print(inputs.shape, targets.shape)
        vv = vis_feature # [B 768]
        la1 = lan_feature[:, :, 0] # [B, 768, 20]

        vv1 = F.normalize(vv, dim=1)
        # la1 = self.adpool(la.unsqueeze(3)).view(la.shape[0], la.shape[1])
        la1 = F.normalize(la1, dim=1)
        score1 = torch.sum(vv1 * la1, dim=1) * self.team
        score11 = torch.exp(score1)
        # pdb.set_trace()
        # mask = torch.ones((self.batch, self.batch)) * (torch.eye(self.batch, self.batch) == 0) #[B, B]
        # mask = mask.cuda()
        score2 = torch.matmul(vv1, la1.transpose(1, 0))
        # pdb.set_trace()
        score2 = score2 * self.team
        # score21 = torch.exp(score2) * mask
        score21 = torch.exp(score2)
        score22 = torch.sum(score21)
        # pdb.set_trace()
        score12 = score11 / score22
        final = torch.sum(-torch.log(score12)) / vis_feature.shape[0]
        # pdb.set_trace()

        return final







