import pdb
from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
from some_functions import lan_cossim_fun
from .lib_functions import Nce_Contrast_Loss

class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.cossim = lan_cossim_fun()
        self.contrastive = Nce_Contrast_Loss()

    def forward(self, x, l_feats, l_feats1, l_mask):

        input_shape = x.shape[-2:]
        l_new, features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        defea, l1, x = self.classifier(l_feats1, x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        if defea.shape[0] > 1:
            loss_contrastive = self.contrastive(defea, l_new)
            loss_sim = self.cossim(l_new, l1, l_mask)
        else:
            loss_contrastive = 0
            loss_sim = 0

        return loss_contrastive, loss_sim, x

class LAVT(_LAVTSimpleDecode):
    pass


class _LAVTSimpleDecodeconloss_endconv(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecodeconloss_endconv, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.cossim = lan_cossim_fun()
        self.contrastive = Nce_Contrast_Loss()

    def forward(self, x, l_feats, l_feats1, l_mask):

        input_shape = x.shape[-2:]
        l_new, features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        defea, l1, x = self.classifier(l_feats1, x_c4, x_c3, x_c2, x_c1, input_shape)
        # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        if defea.shape[0] > 1:
            loss_contrastive = self.contrastive(defea, l_new)
            loss_sim = self.cossim(l_new, l1, l_mask)
        else:
            loss_contrastive = 0
            loss_sim = 0

        return loss_contrastive, loss_sim, x

class LAVT1(_LAVTSimpleDecodeconloss_endconv):
    pass

class _LAVTSimpleDecodeconloss(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecodeconloss, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        l_new, (x_c1, x_c2, x_c3, x_c4) = features
        defea, x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return l_new, defea, x

class LAVTconloss(_LAVTSimpleDecodeconloss):
    pass

class _LAVTSimpleDecode_cycle(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode_cycle, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        pre1, pre2, x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        pre1 = F.interpolate(pre1, size=input_shape, mode='bilinear', align_corners=True)
        pre2 = F.interpolate(pre2, size=input_shape, mode='bilinear', align_corners=True)

        return pre1, pre2, x


class LAVT_cycle(_LAVTSimpleDecode_cycle):
    pass

class _LAVTSimpleDecode_KC(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode_KC, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT_kcdecode(_LAVTSimpleDecode_KC):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass
