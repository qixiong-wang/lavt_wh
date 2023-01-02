import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from torch.nn import functional as F

@torch.no_grad()
def all_gather_tensor(x, gpu=None, save_memory=False):
    
    rank, world_size = get_dist_info()

    if not save_memory:
        # all gather features in parallel
        # cost more GPU memory but less time
        # x = x.cuda(gpu)
        x_gather = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_gather, x, async_op=False)
#         x_gather = torch.cat(x_gather, dim=0)
    else:
        # broadcast features in sequence
        # cost more time but less GPU memory
        container = torch.empty_like(x).cuda(gpu)
        x_gather = []
        for k in range(world_size):
            container.data.copy_(x)
            print("gathering features from rank no.{}".format(k))
            dist.broadcast(container, k)
            x_gather.append(container.cpu())
#         x_gather = torch.cat(x_gather, dim=0)
        # return cpu tensor
    return x_gather

def undefined_l_gather(vis_features,lan_features):
    resized_num = 10000
    pos_num = min(vis_features.size(0),resized_num)
    if vis_features.size(0)>resized_num:
        print(f'{vis_features.size(0)}out of {resized_num}')
    resized_vis_features = torch.empty((resized_num,vis_features.size(1))).to(vis_features.device)
    resized_vis_features[:pos_num,:] = vis_features[:pos_num,:]
    resized_lan_features = torch.empty((resized_num,lan_features.size(1))).to(lan_features.device)
    resized_lan_features[:pos_num] = lan_features[:pos_num,:]
    pos_num = torch.tensor([pos_num]).to(vis_features.device)
    all_pos_num = all_gather_tensor(pos_num)
    all_vis_features = all_gather_tensor(resized_vis_features)
    all_lan_features = all_gather_tensor(resized_lan_features)
    gather_vis_features = []
    gather_lan_features = []
    for index,p_num in enumerate(all_pos_num):
        gather_vis_features.append(all_vis_features[index][:p_num,:])
        gather_lan_features.append(all_lan_features[index][:p_num,:])
    gather_vis_features = torch.cat(gather_vis_features,dim=0)
    gather_lan_features = torch.cat(gather_lan_features,dim=0)
    return gather_vis_features,gather_lan_features



class Memory_queue(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, number_of_instance=200, feat_len=768):
        """
        Args:
            num_persons (int): Number of labeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(Memory_queue, self).__init__()
        self.register_buffer("vis_memory_queue", torch.rand(number_of_instance, feat_len))
        self.register_buffer("lag_memory_queue", torch.rand(number_of_instance, feat_len))
        self.vis_memory_queue=F.normalize(self.vis_memory_queue,dim=1)
        self.lag_memory_queue=F.normalize(self.lag_memory_queue,dim=1)

        self.register_buffer("tail", torch.tensor(0).long())


    def forward(self, vis_feat, lag_feat):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely the similarities
                                             between proposals and labeled persons.
        """
        vis_feat_gather,lag_feat_gather = undefined_l_gather(vis_feat,lag_feat)
        # if vis_feat.get_device() == 0:
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     dist.barrier()
        
        # gather_features,gather_pid_labels = undefined_l_gather(features,pid_labels)
        with torch.no_grad():
            sample_number = vis_feat_gather.shape[0]
            for indx in range(sample_number):
                self.vis_memory_queue[self.tail] = vis_feat_gather[indx]
                self.lag_memory_queue[self.tail] = lag_feat_gather[indx]
                # self.large_batch_queue[label,self.tail[label]] = torch.mean(features[pid_labels==label],dim=0)
                self.tail+=1
                if self.tail >= self.lag_memory_queue.shape[0]:
                    self.tail -= self.lag_memory_queue.shape[0]

        return self.vis_memory_queue, self.lag_memory_queue