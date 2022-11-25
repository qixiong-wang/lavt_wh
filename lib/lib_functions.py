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

        # hidden1 = vv
        # hidden2 = la1
        #
        #
        # hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
        # hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)
        #
        # hidden1_large = hidden1
        # hidden2_large = hidden2
        # labels = torch.arange(0, self.batch).to(device=hidden1.device)
        # masks = torch.nn.functional.one_hot(torch.arange(0, self.batch), num_classes=self.batch).to(
        #     device=hidden1.device, dtype=torch.float)
        #
        # logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / self.team  # shape (batch_size, batch_size)
        # logits_aa = logits_aa - masks * self.LARGE_NUM
        # logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / self.team  # shape (batch_size, batch_size)
        # logits_bb = logits_bb - masks * self.LARGE_NUM
        # logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / self.team  # shape (batch_size, batch_size)
        # logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / self.team  # shape (batch_size, batch_size)
        # pdb.set_trace()
        #
        # loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        # loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        # lossf = loss_a + loss_b


        # score = torch.matmul(vv1, la1.permute(1, 0))





        # vv1 = F.normalize(vv, dim=1)
        # la1 = self.adpool(la.unsqueeze(3)).view(la.shape[0], la.shape[1])
        # la1 = F.normalize(la1, dim=1)
        # score1 = torch.sum(vv1 * la1, dim=1) * self.team
        # score11 = torch.exp(score1)
        # # pdb.set_trace()
        # # mask = torch.ones((self.batch, self.batch)) * (torch.eye(self.batch, self.batch) == 0) #[B, B]
        # # mask = mask.cuda()
        # score2 = torch.matmul(vv1, la1.transpose(1, 0))
        # # pdb.set_trace()
        # score2 = score2 * self.team
        # # score21 = torch.exp(score2) * mask
        # score21 = torch.exp(score2)
        # score22 = torch.sum(score21)
        # # pdb.set_trace()
        # score12 = score11 / score22
        # final = torch.sum(-torch.log(score12)) / vis_feature.shape[0]
        # # pdb.set_trace()
        #
        # return final







