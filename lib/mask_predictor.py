import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


# Copyright (c) OpenMMLab. All rights reserved.
from tkinter.tix import Tree
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import ModuleList
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask = None,
                     query_pos= None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask = None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask= None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos  = None,
                     query_pos = None):


        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask= None,
                    pos= None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory):
        if self.normalize_before:
            return self.forward_pre(tgt, memory)
        return self.forward_post(tgt, memory)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)
        self.conv_emb = nn.Conv2d(hidden_size, 768, 1)
        self.num_classes = 2
        self.text_emb = nn.Parameter(torch.randn(1, 1, 768))

        embed_dims = 768
        num_heads = 2
        self.transformer = torch.nn.Transformer(d_model=embed_dims, nhead=num_heads, batch_first=True)
        # self.transformer_cross_attention_layers = CrossAttentionLayer(
        #         d_model=embed_dims,
        #         nhead=num_heads,
        #         dropout=0.0,
        #         normalize_before=False,
        #     )
        # self.transformer_self_attention_layers= SelfAttentionLayer(
        #             d_model=embed_dims,
        #             nhead=num_heads,
        #             dropout=0.0,
        #             normalize_before=False,
        #         )

        # self.transformer_ffn_layers = FFNLayer(
        #         d_model=embed_dims,
        #         dim_feedforward=num_heads,
        #         dropout=0.0,
        #         normalize_before=False,
        #     )

    def forward(self, x_c4, x_c3, x_c2, x_c1, target=None):

        # fuse Y4 and Y3

        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
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
        x = torch.cat([x, x_c1], dim=1)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        output = self.conv1_1(x)

        vis_embedding = self.conv_emb(x)

        vis_embedding = F.interpolate(input=vis_embedding, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        b, c, h, w = vis_embedding.shape
        vis_embedding = vis_embedding.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        cls_seg_feat = self.text_emb.expand(output.size(0), -1, -1)
        text_embedding = self.transformer(vis_embedding,cls_seg_feat)

        # text_embedding = self.transformer_cross_attention_layers(cls_seg_feat,vis_embedding)
        # text_embedding = self.transformer_self_attention_layers(text_embedding)
        # text_embedding = self.transformer_ffn_layers(text_embedding)

        # if target!=None:
        #     target = target.unsqueeze(1).float()
        #     target = F.interpolate(input=target,size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        #     embedding = torch.sum(torch.multiply(target,embedding),dim=[2,3])/torch.sum(torch.sum(target,dim=[2,3]))
        
        # else:
            # text_embedding = F.adaptive_avg_pool2d(vis_embedding,output_size=1)


        # output = x

        # b, c, h, w = output.shape
        # output = output.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        # cls_seg_feat = self.cls_emb.expand(output.size(0), -1, -1)
        # cls_seg_feat = self.transformer_cross_attention_layers(cls_seg_feat,output)
        # cls_seg_feat = self.transformer_self_attention_layers(cls_seg_feat)
        # cls_seg_feat = self.transformer_ffn_layers(cls_seg_feat)

        # output = self.patch_proj(output)
        # cls_seg_feat = self.classes_proj(cls_seg_feat)

        # output = F.normalize(output, dim=2, p=2)
        # cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)
        
        # output = output @ cls_seg_feat.transpose(1, 2)
        # output = self.mask_norm(output)
        # output = output.permute(0, 2, 1).contiguous().view(b,-1, h, w)
        return output,text_embedding
