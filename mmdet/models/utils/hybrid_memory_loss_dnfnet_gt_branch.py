# Ge et al. Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.  # noqa
# Written by Yixiao Ge.

import imp
from re import L
from this import d
import torch
import torch.nn.functional as F
from torch import autograd, nn
import collections
import numpy as np
import os

from mmdet.utils import all_gather_tensor
import sys
import numpy as np
sys.path.append("/home/linhuadong/DNFNet/mmdet/models/utils")
# import min_search_cuda
# import max_search_cuda
from torch.cuda.amp import custom_fwd, custom_bwd
from itertools import accumulate


class HM_part(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, part_inputs, gt_inputs, gt_part_inputs, indexes, features, part_features, IoU, momentum, 
                    IoU_memory_clip, update_flag, update_method):
        
        ctx.features = features
        ctx.part_features = part_features
        ctx.momentum = momentum
        ctx.update_method = update_method   # Update methods, such as momentum updates
        ctx.update_flag = update_flag   # If the greedy update strategy is adopted, the features that need to be updated
        ctx.IoU_memory_clip = IoU_memory_clip
        
        outputs = inputs.mm(ctx.features.t())
        part_outputs = part_inputs.mm(ctx.part_features.t())
        gt_outputs = gt_inputs.mm(ctx.features.t())
        gt_part_outputs = gt_part_inputs.mm(ctx.part_features.t())

        # all_inputs = all_gather_tensor(inputs)
        # part_inputs = all_gather_tensor(part_inputs)
        all_gt_inputs = all_gather_tensor(gt_inputs)
        all_gt_part_inputs = all_gather_tensor(gt_part_inputs)
        all_indexes = all_gather_tensor(indexes)
        all_IoU = all_gather_tensor(IoU)
        
        ctx.save_for_backward(all_gt_inputs, all_gt_part_inputs, all_indexes, all_IoU)
        return outputs, part_outputs, gt_outputs, gt_part_outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs, grad_part_outputs, grad_gt_outputs, grad_gt_part_outputs):
        gt_inputs, gt_part_inputs, indexes, IoU = ctx.saved_tensors
        grad_inputs, grad_part_inputs, grad_gt_inputs, grad_gt_part_inputs = None, None, None, None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
            grad_part_inputs = grad_part_outputs.mm(ctx.part_features)
            grad_gt_inputs = grad_gt_outputs.mm(ctx.features)
            grad_gt_part_inputs = grad_gt_part_outputs.mm(ctx.part_features)
        IoU = torch.clamp(IoU, min=ctx.IoU_memory_clip[0], max=ctx.IoU_memory_clip[1])
        # Update the memory bank using the gt branch feature
        for x, px, y, iou, uf in zip(gt_inputs, gt_part_inputs, indexes, IoU, ctx.update_flag):
            if ctx.update_method == "momentum":
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.part_features[y] = ctx.momentum * ctx.part_features[y] + (1.0 - ctx.momentum) * px
            elif ctx.update_method == "iou":
                ctx.features[y] = (1 - iou) * ctx.features[y] + iou * x
                ctx.part_features[y] = (1 - iou) * ctx.part_features[y] + iou * px
            elif ctx.update_method == "max_iou":
                if uf: 
                    ctx.features[y] = x
                    ctx.part_features[y] = px
            elif ctx.update_method == "max_iou_momentum":
                if uf: 
                    ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                    ctx.part_features[y] = ctx.momentum * ctx.part_features[y] + (1.0 - ctx.momentum) * px
            
            ctx.features[y] /= ctx.features[y].norm()
            ctx.part_features[y] /= ctx.part_features[y].norm()

        return grad_inputs, grad_part_inputs, grad_gt_inputs, grad_gt_part_inputs, None, None, None, None, None, None, None, None

def hm_part(inputs, part_inputs, gt_inputs, gt_part_inputs, indexes, features, part_features, IoU, momentum, IoU_memory_clip, \
                    update_flag, update_method):
    return HM_part.apply(
        inputs, part_inputs, gt_inputs, gt_part_inputs, indexes, features, part_features, IoU, torch.Tensor([momentum]).to(inputs.device), \
            torch.Tensor(IoU_memory_clip).to(inputs.device), update_flag, update_method
    )

class HM(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, indexes, features, IoU, update_method, momentum, update_flag, IoU_memory_clip):
        ctx.features = features
        ctx.momentum = momentum
        ctx.update_method = update_method
        ctx.update_flag = update_flag
        ctx.IoU_memory_clip = IoU_memory_clip

        outputs = inputs.mm(ctx.features.t())

        all_inputs = all_gather_tensor(inputs)
        all_indexes = all_gather_tensor(indexes)
        all_IoU = all_gather_tensor(IoU)

        ctx.save_for_backward(all_inputs, all_indexes, all_IoU)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, indexes, IoU = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        
        IoU = torch.clamp(IoU, min=ctx.IoU_memory_clip[0], max=ctx.IoU_memory_clip[1])
        for x, y, iou, uf in zip(inputs, indexes, IoU, ctx.update_flag):
            if ctx.update_method == "momentum":
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            elif ctx.update_method == "iou":
                ctx.features[y] = (1 - iou) * ctx.features[y] + iou * x
            elif ctx.update_method == "max_iou":
                if uf: ctx.features[y] = x
            elif ctx.update_method == "max_iou_momentum":
                if uf: ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None, None, None, None

def hm(inputs, indexes, features, IoU, update_method=None, momentum=0.5, update_flag=None, IoU_memory_clip=0.2):
    return HM.apply(
        inputs, indexes, features, IoU, update_method, torch.Tensor([momentum]).to(inputs.device), update_flag, torch.Tensor(IoU_memory_clip).to(inputs.device), 
    
    )

class HybridMemoryMultiFocalPercentDnfnetGtBranch(nn.Module):

    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, cluster_top_percent=0.1, use_cluster_hard_loss=True, testing=False,
                    IoU_memory_clip=[0.2, 0.9], use_part_feat=False, update_method="max_iou", cluster_mean_method="time_consistency", tc_winsize=500):
        super(HybridMemoryMultiFocalPercentDnfnetGtBranch, self).__init__()
        self.use_cluster_hard_loss = use_cluster_hard_loss
        self.num_features = num_features
        self.IoU_memory_clip = IoU_memory_clip
        self.use_part_feat = use_part_feat
        self.update_method = update_method
        self.cluster_mean_method = cluster_mean_method
        self.tc_winsize = tc_winsize
        self.clock = 0

        if testing == True:
            num_memory = 500
        self.num_memory = num_memory
        self.momentum = momentum
        self.temp = temp

        # for mutli focal
        self.cluster_top_percent = cluster_top_percent

        self.idx = torch.zeros(num_memory).long()
        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
        self.register_buffer("tflag", torch.zeros(num_memory).long())
        if self.use_part_feat:
            self.register_buffer("part_features", torch.zeros(num_memory, num_features))

    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx[:ids.shape[0]].data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _init_tflag(self):
        self.tflag.data.copy_(torch.zeros_like(self.tflag).long()).to(self.labels.device)

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features[:features.shape[0]].data.copy_(features.float().to(self.features.device))
        
    @torch.no_grad()
    def _update_part_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.part_features[:features.shape[0]].data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))
    
    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    @torch.no_grad()
    def get_features(self, indexes):
        return self.features[indexes].clone()

    @torch.no_grad()
    def get_all_features(self):
        return self.features.clone()

    @torch.no_grad()
    def get_all_cluster_ids(self):
        return self.labels.clone()

    def masked_softmax_multi_focal(self, vec, mask, dim=1, targets=None, epsilon=1e-6):
        """
            :vec: [B, u], 与聚类中心的相似度
            :mask: [B, u], 某些簇的数量为0
            :targets: [B, 1]
            :labels: [N, ]
        """
        exps = torch.exp(vec)   # [B, u]
        one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=exps.shape[1]) # [B, u]
        one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)  # 返回一个与size大小相同的用1填充的张量
        one_hot_neg = one_hot_neg - one_hot_pos # 负样本的one_hot
        masked_exps = exps * mask.float().clone()   # 与负聚类中心的相似度
        
        # 难样本挖掘
        neg_exps = exps.new_zeros(size=exps.shape)
        neg_exps[one_hot_neg>0] = masked_exps[one_hot_neg>0]
        ori_neg_exps = neg_exps

        neg_exps = neg_exps / neg_exps.sum(dim=1, keepdim=True) # 难样本归一化

        new_exps = masked_exps.new_zeros(size=exps.shape)
        new_exps[one_hot_pos>0] = masked_exps[one_hot_pos>0]
    
        sorted, indices = torch.sort(neg_exps, dim=1, descending=True)  # 排序得到相似度最大(难度最大)的负样本
        sorted_cum_sum = torch.cumsum(sorted, dim=1)
        sorted_cum_diff = (sorted_cum_sum - self.cluster_top_percent).abs()
        sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)  # 获得K的大小
        min_values = sorted[torch.range(0, sorted.shape[0]-1).long(), sorted_cum_min_indices]   # 获取K对应的val
        min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)   # 前面除neg_exps.sum(),所以这里乘回去
        ori_neg_exps[ori_neg_exps<min_values] = 0   # 小于阈值,即难度稍微小的负样本不考虑

        new_exps[one_hot_neg>0] = ori_neg_exps[one_hot_neg>0]   # 做分母
        masked_exps = new_exps
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return masked_exps / masked_sums    # softmax

    def get_hard_cluster_loss(self, labels, cluster_inputs, targets):
        """
            :cluster_inputs: [B, N], Similarity between the sample and the cluster center
            :targets: [B]
            :labels: [N]
            :IoU: [N]
        """
        B = cluster_inputs.shape[0]

        # Local window
        if self.cluster_mean_method == "time_consistency":
            tflag_latest = (self.clock - self.tflag) < self.tc_winsize  # [N]
            cluster_inputs = cluster_inputs[:, tflag_latest == True]    # [B, N']
            labels = labels[tflag_latest == True]   # [N']
            # if tflag_latest.sum() < 500:
            #     import ipdb;    ipdb.set_trace()

        self.num_memory = labels.shape[0]
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) # [C], 求每一个簇样本个数
        mask = (nums > 0).float()

        sim = torch.zeros(labels.max() + 1, B).float().cuda() # [C, B], C = labels.max() + 1 indicates the number of labels    
        sim.index_add_(0, labels, cluster_inputs.t().contiguous())  # Each column represents the similarity between instance and cluster
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) # average features in each cluster, C * B

        # weight = torch.load(os.path.join('saved_file', 'weight.pth')).cuda()    # [N]
        # weighted_cluster_inputs = cluster_inputs * weight[None]
        # sim = torch.zeros(labels.max() + 1, B).float().cuda() # C * B, unique label num: C = labels.max() + 1表示标签的数量
        # sim.index_add_(0, labels, weighted_cluster_inputs.t().contiguous())  # 每一列表示minibatch中instance与同一个簇中所有instance的相似度的和
        # 求方差
        # cluster_center = self.get_cluster_centroid()    # [C, 256]
        # average_std, average_std_exclude_outliers = self.get_gaussion_distribution(cluster_center)

        mask = mask.expand_as(sim)
        masked_sim = self.masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets) # sim: u * B, mask:u * B, masked_sim: B * u
        cluster_hard_loss = F.nll_loss(torch.log(masked_sim + 1e-6), targets, reduce=False)
        cluster_hard_loss = cluster_hard_loss.mean()
        return cluster_hard_loss#, average_std.detach(), average_std_exclude_outliers.detach()

    def get_gaussion_distribution(self, cluster_center):
        # 求簇的方差
        # import ipdb;    ipdb.set_trace()
        dist = self.euclidean_dist(cluster_center, self.features)   # [C, N]
        dist_mask = torch.arange(cluster_center.shape[0])[:, None].cuda() == self.labels
        dist *= dist_mask
        cluster_size = dist_mask.sum(dim=-1)
        cluster_size[cluster_size == 0] = 1.    # 不可能存在0的情况
        std = dist.sum(dim=-1) / cluster_size
        average_std = std.mean()
        # 除去异常点的情况
        outliers_mask = (cluster_size > 1)
        average_std_exclude_outliers = (std * outliers_mask).sum() / outliers_mask.sum()
        return average_std, average_std_exclude_outliers

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def get_cluster_centroid(self):
        nums = torch.zeros(self.labels.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, self.labels, torch.ones(self.labels.shape[0], 1).float().cuda()) # C * 1
        mask = (nums > 0).float()
        cluster_center = torch.zeros(self.labels.max() + 1, self.features.shape[-1]).float().cuda()  # [C, 256]
        cluster_center.index_add_(0, self.labels, self.features.contiguous())
        cluster_center /= (mask * nums + (1 - mask)).clone().expand_as(cluster_center)
        return cluster_center

    def get_iou_loss(self, feats, targets):
        """
            :feats: [B, 256]
            :iou_target: [B]
            每个样本和IoU最好的样本拉近
        """
        cos_loss = 1 - feats.mm(feats.t())   # [B, B]
        iou_loss = cos_loss[range(len(targets)), targets].mean()

        return iou_loss
        
    def get_update_flag(self, indexes, IoU):
        """
            indexes: 一张图片中对应同一个person的编号
        """
        unique_indexes = torch.unique(indexes)
        update_flag = torch.zeros_like(indexes).bool().to(indexes.device)
        iou_target = torch.zeros_like(IoU).long().to(indexes.device)
        for i, uid in enumerate(unique_indexes):
            IoU_tmp = IoU.clone()
            IoU_tmp[indexes!=uid] = -1
            maxid = torch.argmax(IoU_tmp)
            update_flag[maxid] = True
            iou_target[indexes==uid] = maxid
        return update_flag, iou_target

    def get_m2o_loss(self, feats, targets, pos_is_gt_list):
        """
            每张图片中的样本和gt proposal拉进
        """
        proposals_nums = [len(value) for value in pos_is_gt_list]
        gt_nums = [torch.sum(value).item() for value in pos_is_gt_list]
        pred_nums = [proposals_nums[i] - gt_nums[i] for i in range(len(proposals_nums))]
        cumsum_pro_nums = list(accumulate([0] + proposals_nums))

        m2o_loss = torch.tensor(0.).cuda()
        for i in range(1, len(cumsum_pro_nums)):
            gt_num = gt_nums[i-1]
            pred_num = pred_nums[i-1]
            if gt_num > 0 and pred_num > 0:
                gt_targets = targets[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==1]
                pred_targets = targets[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==0]
                
                gt_feats = feats[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==1]
                pred_feats = feats[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==0]
                
                cos_loss = 1 - gt_feats.mm(pred_feats.t())
                flag = gt_targets[:, None] == pred_targets[None]
                m2o_sim = cos_loss[flag].mean()
                m2o_loss += m2o_sim
        
        m2o_loss /= len(proposals_nums)
        return m2o_loss

    def forward(self, feats, part_feats, gt_feats, gt_part_feats, indexes, IoU, top_IoU, bottom_IoU):
        """
            The memory bank stores the gt branch feature
            :feats: [B, 256]
            :indexes: [B, ]
            :IoU: [B, ]
        """
        self.clock += 1
        self.tflag[indexes] = self.clock

        update_flag, iou_target = self.get_update_flag(indexes, IoU)
        # top_update_flag, top_iou_target = self.get_update_flag(indexes, top_IoU)
        # bottom_update_flag, bottom_iou_target = self.get_update_flag(indexes, bottom_IoU)

        losses = {}
        targets = self.labels[indexes].clone()
        labels = self.labels.clone()
        feats = F.normalize(feats, p=2, dim=1)
        gt_feats = F.normalize(gt_feats, p=2, dim=1)
        if self.use_part_feat:
            part_feats = F.normalize(part_feats, p=2, dim=1)
            gt_part_feats = F.normalize(gt_part_feats, p=2, dim=1)
            inputs, part_inputs, gt_inputs, gt_part_inputs = hm_part(feats, part_feats, gt_feats, gt_part_feats, indexes, \
                                    self.features, self.part_features, IoU, self.momentum, self.IoU_memory_clip, \
                                    update_flag, self.update_method)   # [B, N]
            inputs, part_inputs, gt_inputs, gt_part_inputs = inputs / self.temp, part_inputs / self.temp, \
                                                            gt_inputs / self.temp, gt_part_inputs / self.temp
        else:
            inputs = hm(feats, indexes, self.features, IoU, self.update_method, self.momentum, update_flag, self.IoU_memory_clip)   # [B, N]
            inputs /= self.temp

        if self.use_cluster_hard_loss:
            losses["global_cluster_loss"] = torch.tensor(0.)
            losses["part_cluster_loss"] = torch.tensor(0.)
            if targets.shape[0] > 0:
                losses["global_cluster_loss"] = self.get_hard_cluster_loss(labels.clone(), inputs, targets)
                losses["gt_global_cluster_loss"] = self.get_hard_cluster_loss(labels.clone(), gt_inputs, targets)
                if self.use_part_feat:
                    losses["part_cluster_loss"] = self.get_hard_cluster_loss(labels.clone(), part_inputs, targets)
                    losses["gt_part_cluster_loss"] = self.get_hard_cluster_loss(labels.clone(), gt_part_inputs, targets)
        return losses