from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
import torchvision.transforms as T

class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier1, classifier2):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier1 = classifier1
        self.classifier2 = classifier2

    def forward(self, x, l_feats, l_mask, gt=None):
        input_shape = x.shape[-2:]
        h,w=input_shape[0],input_shape[1]

        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        x_c1_ms = F.interpolate(input=x_c1, scale_factor=1.25, mode='bilinear', align_corners=True)
        x_c2_ms = F.interpolate(input=x_c2, scale_factor=1.25, mode='bilinear', align_corners=True)
        x_c3_ms = F.interpolate(input=x_c3, scale_factor=1.25, mode='bilinear', align_corners=True)
        x_c4_ms = F.interpolate(input=x_c4, scale_factor=1.25, mode='bilinear', align_corners=True)

        x1 = self.classifier1(x_c4, x_c3, x_c2, x_c1)

        x2 = self.classifier2(x_c4_ms, x_c3_ms, x_c2_ms, x_c1_ms)
        
        x1 = F.interpolate(x1, size=input_shape, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=input_shape, mode='bilinear', align_corners=True)


        return x1, x2

        # return primary_result

class LAVT(_LAVTSimpleDecode):
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
