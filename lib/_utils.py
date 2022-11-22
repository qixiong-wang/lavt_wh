import pdb
from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
from .draw_attention import visulize_attention1


# class _LAVTSimpleDecode(nn.Module):
#     def __init__(self, backbone, classifier):
#         super(_LAVTSimpleDecode, self).__init__()
#         self.backbone = backbone
#         self.classifier = classifier
#
#     def forward(self, id_num, x, l_feats, l_mask):
#
#         img0 = x
#         input_shape = x.shape[-2:]
#         features = self.backbone(x, l_feats, l_mask)
#         x_c1, x_c2, x_c3, x_c4 = features
#         x = self.classifier(x_c4, x_c3, x_c2, x_c1)
#         x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
#         visulize_attention1(id_num, img0, x)
#
#         return x

class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):

        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x

class LAVT(_LAVTSimpleDecode):
    pass


class _LAVTSimpleDecode_vismatrix(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode_vismatrix, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask, file_name, sentence):

        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask, file_name, sentence)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x

class LAVTvm(_LAVTSimpleDecode_vismatrix):
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
        # pdb.set_trace()

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
