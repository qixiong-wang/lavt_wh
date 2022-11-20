import torch.nn as nn
import pdb

class simple_lan_transformer(nn.Module):
    def __init__(self, hidden_size, lan_size):
        super(simple_lan_transformer, self).__init__()

        self.transformer_model = nn.Transformer(d_model=768, nhead=6, num_encoder_layers=2, num_decoder_layers=2)
        self.conv1 = nn.Conv2d(hidden_size, lan_size, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(lan_size)
        self.relu1 = nn.ReLU()

    def forward(self, vis, lan):

        vis = self.conv1(vis)
        vis = self.bn1(vis)
        vis = self.relu1(vis)
        vis = vis.view(vis.shape[0], vis.shape[1], -1)
        vis = vis.permute(2, 0, 1)
        lan = lan.permute(2, 0, 1)
        out = self.transformer_model(vis, lan)
        out = out.permute(1, 2, 0)

        return out


