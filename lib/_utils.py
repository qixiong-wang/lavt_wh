from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
import torchvision.transforms as T

class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, refinement):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.refinement = refinement
        # self.primary_result_bn = nn.BatchNorm2d(2)

    def forward(self, x, l_feats, l_mask, gt=None):
        input_shape = x.shape[-2:]
        h,w=input_shape[0],input_shape[1]

        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        # x_c1_ms = F.interpolate(input=x_c1, scale_factor=1.25, mode='bilinear', align_corners=True)
        # x_c2_ms = F.interpolate(input=x_c2, scale_factor=1.25, mode='bilinear', align_corners=True)
        # x_c3_ms = F.interpolate(input=x_c3, scale_factor=1.25, mode='bilinear', align_corners=True)
        # x_c4_ms = F.interpolate(input=x_c4, scale_factor=1.25, mode='bilinear', align_corners=True)

        primary_result = self.classifier(x_c4, x_c3, x_c2, x_c1)
        
        # x_ms = self.classifier(x_c4_ms, x_c3_ms, x_c2_ms, x_c1_ms)
        
        primary_result = F.interpolate(primary_result, size=input_shape, mode='bilinear', align_corners=True)
        primary_result_p = F.softmax(primary_result, dim=1)
        # primary_result_p = self.primary_result_bn(primary_result)

        refine_result = self.refinement(torch.cat((x,primary_result_p),dim=1))

        # if gt==None:
        #     gt = F.softmax(primary_result, dim=1)
        #     gt = torch.argmax(gt,dim=1)
        # try:
        #     target_images = []
        #     target_positions = []
        #     for i in range(gt.shape[0]):
        #         target_position= torch.where(gt[i]==1)
        #         y1, y2 = torch.min(target_position[0]),torch.max(target_position[0])
        #         x1, x2 = torch.min(target_position[1]),torch.max(target_position[1])
        #         extend_h = max(10,int((y2-y1)/10))
        #         extend_w = max(10,int((x2-x1)/10))
        #         extend_x1 = max(0,x1-extend_w)
        #         extend_x2 = min(w,x2+extend_w)
        #         extend_y1 = max(0,y1-extend_h)
        #         extend_y2 = min(h,y2+extend_h)
        #         target_positions.append([extend_x1,extend_x2,extend_y1,extend_y2])

        #         target_img = x[i][:,extend_y1:extend_y2,extend_x1:extend_x2]
        #         target_img = F.interpolate(target_img.unsqueeze(0), size=input_shape, mode='bilinear', align_corners=True)
        #         target_images.append(target_img)
        #     target_images = torch.cat(target_images,dim=0)
        #     target_features = self.refinement(target_images)
        #     refine_result = primary_result.clone().detach()
            
        #     for i in range(gt.shape[0]):
        #         extend_x1,extend_x2,extend_y1,extend_y2 = target_positions[i]
        #         refine_result[i][:,extend_y1:extend_y2,extend_x1:extend_x2] = F.interpolate(target_features[i].unsqueeze(0),size=(extend_y2-extend_y1,extend_x2-extend_x1), mode='bilinear', align_corners=True).squeeze()

        # refine_result = torch.zeros_like(primary_result)
        refine_result = F.interpolate(refine_result, size=input_shape, mode='bilinear', align_corners=True)

        return primary_result, refine_result

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
