from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel

from .memory_queue import Memory_queue



class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.number_of_instance =200
        self.memory_queue = Memory_queue(number_of_instance=200, feat_len=768)
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask, target=None):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x, vis_embedding = self.classifier(x_c4, x_c3, x_c2, x_c1, target)

        vis_embedding = vis_embedding.squeeze()

        if len(vis_embedding.shape)>1: ### eval
            vis_embedding = F.normalize(vis_embedding,dim=1)
            batch_size=l_mask.shape[0]
            l_feat_last = []
            for i in range(batch_size):
                l_feat_last.append(l_feats[i,:,torch.where(l_mask[i]==1)[0][-1]])
            l_feat_last = torch.stack(l_feat_last)
            l_feat_last = F.normalize(l_feat_last,dim=1)

            # vis_embedding_queue, l_feat_queue = self.memory_queue(vis_embedding,l_feat_last)

            contrast_label = torch.eye(batch_size).cuda(l_feat_last.device)
            # img_text_logits = F.softmax(10*torch.matmul(vis_embedding_queue,l_feat_last.permute(1,0)),dim=0)
            # text_img_logits = F.softmax(10*torch.matmul(l_feat_queue,vis_embedding.permute(1,0)),dim=0)
            img_text_logits = F.softmax(10*torch.matmul(vis_embedding,l_feat_last.permute(1,0)),dim=0)
            text_img_logits = F.softmax(10*torch.matmul(l_feat_last,vis_embedding.permute(1,0)),dim=0)

            pos_ind = torch.arange(batch_size).cuda(l_feat_last.device) + self.memory_queue.tail - batch_size
            pos_ind = torch.where(pos_ind<0,pos_ind+self.number_of_instance,pos_ind)
            # loss_recon = -torch.multiply(contrast_label,torch.log(img_text_logits[pos_ind]))-torch.multiply(contrast_label,torch.log(text_img_logits[pos_ind]))
            # if self.memory_queue.tail==198:
            #     import pdb
            #     pdb.set_trace()
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
