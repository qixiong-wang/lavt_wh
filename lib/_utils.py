from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel





class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x, embedding = self.classifier(x_c4, x_c3, x_c2, x_c1)
        embedding = embedding.squeeze()
        if len(embedding.shape)>1: ### eval
            embedding = F.normalize(embedding,dim=1)
            batch_size=l_mask.shape[0]
            l_feat_last = []
            for i in range(batch_size):
                l_feat_last.append(l_feats[i,:,torch.where(l_mask[i]==1)[0][-1]])
            l_feat_last = torch.stack(l_feat_last)
            l_feat_last = F.normalize(l_feat_last,dim=1)

            contrast_label = torch.eye(batch_size).cuda(l_feat_last.device)
            img_text_logits = F.softmax(torch.matmul(embedding,l_feat_last.permute(1,0)),dim=1)
            text_img_logits = F.softmax(torch.matmul(embedding,l_feat_last.permute(1,0)),dim=0)
            loss_recon = -torch.multiply(contrast_label,torch.log(img_text_logits))-torch.multiply(contrast_label,torch.log(text_img_logits))
            loss_recon = torch.mean(loss_recon)
        else:
            loss_recon = 0
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x, loss_recon

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
