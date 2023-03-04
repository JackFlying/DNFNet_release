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

def get_mean_conv(input_vec):
    mean = torch.mean(input_vec, axis=0)
    x = input_vec - mean
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0] - 1 if x.shape[0] > 1 else 1)
    return mean[None], cov_matrix[None]

class HM_part(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, bottom_inputs, top_inputs, indexes, labels, cluster_mean, cluster_std, features, bottom_features, top_features, mIoU, \
                    IoU, top_IoU, bottom_IoU, momentum, IoU_momentum, IoU_memory_clip, update_flag, top_update_flag, \
                        bottom_update_flag, update_method, targets):
        ctx.cluster_mean = cluster_mean
        ctx.cluster_std = cluster_std
        ctx.labels = labels
        ctx.momentum = momentum
        ctx.mIoU = mIoU
        ctx.IoU_momentum = IoU_momentum
        ctx.features = features
        ctx.bottom_features = bottom_features
        ctx.top_features = top_features
        ctx.IoU_memory_clip = IoU_memory_clip
        ctx.update_flag = update_flag
        ctx.top_update_flag = top_update_flag
        ctx.bottom_update_flag = bottom_update_flag
        ctx.update_method = update_method
        ctx.targets = targets

        z = torch.randn(256)[None, :, None].cuda()
        ctx.sample = ctx.cluster_mean + (ctx.cluster_std @ z).squeeze(-1)
        outputs = inputs.mm(ctx.sample.t())
        bottom_outputs = bottom_inputs.mm(ctx.bottom_features.t())
        top_outputs = top_inputs.mm(ctx.top_features.t())

        all_inputs = all_gather_tensor(inputs)
        all_indexes = all_gather_tensor(indexes)
        all_IoU = all_gather_tensor(IoU)
        all_top_IoU = all_gather_tensor(top_IoU)
        all_bottom_IoU = all_gather_tensor(bottom_IoU)
        bottom_inputs = all_gather_tensor(bottom_inputs)
        top_inputs = all_gather_tensor(top_inputs)
        IoU_memory_clip = all_gather_tensor(IoU_memory_clip)
        update_flag = all_gather_tensor(update_flag)
        top_update_flag = all_gather_tensor(top_update_flag)
        bottom_update_flag = all_gather_tensor(bottom_update_flag)
        update_method = all_gather_tensor(update_method)
        targets = all_gather_tensor(targets)
        
        ctx.save_for_backward(all_inputs, all_indexes, all_IoU, all_top_IoU, all_bottom_IoU, bottom_inputs, \
                top_inputs, IoU_memory_clip, update_flag, top_update_flag, bottom_update_flag, targets)
        return outputs, bottom_outputs, top_outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs, grad_bottom_outputs, grad_top_outputs):
        inputs, indexes, IoU, top_IoU, bottom_IoU, bottom_inputs, top_inputs, IoU_memory_clip, update_flag, \
                    top_update_flag, bottom_update_flag, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.sample)
            grad_bottom_outputs = grad_bottom_outputs.mm(ctx.bottom_features)
            grad_top_outputs = grad_top_outputs.mm(ctx.top_features)
        IoU = torch.clamp(IoU, min=IoU_memory_clip[0], max=IoU_memory_clip[1])
        bottom_IoU = torch.clamp(bottom_IoU, min=IoU_memory_clip[0], max=IoU_memory_clip[1])
        top_IoU = torch.clamp(top_IoU, min=IoU_memory_clip[0], max=IoU_memory_clip[1])

        for x, y, b, t, iou, biou, tiou, uf, tuf, buf, tg in zip(inputs, indexes, bottom_inputs, top_inputs, IoU, bottom_IoU, top_IoU,\
                        update_flag, top_update_flag, bottom_update_flag, targets):
            if ctx.update_method == "momentum":
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.bottom_features[y] = ctx.momentum * ctx.bottom_features[y] + (1.0 - ctx.momentum) * b
                ctx.top_features[y] = ctx.momentum * ctx.top_features[y] + (1.0 - ctx.momentum) * t
            elif ctx.update_method == "iou":
                ctx.features[y] = (1 - iou) * ctx.features[y] + iou * x
                ctx.bottom_features[y] = (1 - biou) * ctx.bottom_features[y] + biou * b
                ctx.top_features[y] = (1 - tiou) * ctx.top_features[y] + tiou * t
            elif ctx.update_method == "max_iou":
                if uf: ctx.features[y] = x
                if buf: ctx.bottom_features[y] = b
                if tuf: ctx.top_features[y] = t
            
            ctx.features[y] /= ctx.features[y].norm()
            ctx.bottom_features[y] /= ctx.bottom_features[y].norm()
            ctx.top_features[y] /= ctx.top_features[y].norm()
            ctx.cluster_mean[tg], ctx.cluster_std[tg] = get_mean_conv(ctx.features[ctx.labels == tg])
            
        return grad_inputs, grad_bottom_outputs, grad_top_outputs, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def hm_part(inputs, bottom_inputs, top_inputs, indexes, labels, cluster_mean, cluster_std, features, bottom_features, top_features, mIoU, momentum, IoU_momentum, \
                    IoU, top_IoU, bottom_IoU, IoU_memory_clip, update_flag, top_update_flag, bottom_update_flag, update_method, targets):
    return HM_part.apply(
        inputs, bottom_inputs, top_inputs, indexes, labels, cluster_mean, cluster_std, features, bottom_features, top_features, mIoU, IoU, top_IoU, bottom_IoU, \
        torch.Tensor([momentum]).to(inputs.device), torch.Tensor([IoU_momentum]).to(inputs.device), torch.Tensor(IoU_memory_clip).to(inputs.device), 
        update_flag, top_update_flag, bottom_update_flag, update_method, targets
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
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None, None, None, None

def hm(inputs, indexes, features, IoU, update_method=None, momentum=0.5, update_flag=None, IoU_memory_clip=0.2):
    return HM.apply(
        inputs, indexes, features, IoU, update_method, torch.Tensor([momentum]).to(inputs.device), update_flag, torch.Tensor(IoU_memory_clip).to(inputs.device), 
    
    )

class HybridMemoryMultiFocalPercentCluster(nn.Module):

    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, cluster_top_percent=0.1, instance_top_percent=1, \
                    use_cluster_hard_loss=True, use_instance_hard_loss=False, use_hybrid_loss=False, testing=False, use_uncertainty_loss=False,
                    use_IoU_loss=False, use_IoU_memory=False, IoU_loss_clip=[0.7, 1.0], IoU_memory_clip=[0.2, 0.9], IoU_momentum=0.2,
                    use_part_feat=False, co_learning=False, use_hard_mining=False, use_max_IoU_bbox=False, update_method=None):
        super(HybridMemoryMultiFocalPercentCluster, self).__init__()
        self.use_cluster_hard_loss = use_cluster_hard_loss
        self.use_instance_hard_loss = use_instance_hard_loss
        self.use_hybrid_loss = use_hybrid_loss
        self.num_features = num_features
        self.use_IoU_loss = use_IoU_loss
        self.use_IoU_memory = use_IoU_memory
        self.IoU_loss_clip = IoU_loss_clip
        self.IoU_memory_clip = IoU_memory_clip
        self.IoU_momentum = IoU_momentum
        self.use_part_feat = use_part_feat
        self.update_method = update_method
        self.use_cluster_memory = True

        if testing == True:
            num_memory = 500
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        #for mutli focal
        self.cluster_top_percent = cluster_top_percent
        self.instance_top_percent = instance_top_percent
        self.co_learning = co_learning
        self.use_uncertainty_loss = use_uncertainty_loss
        self.hard_mining = use_hard_mining
        self.use_max_IoU_bbox = use_max_IoU_bbox
        self.iou_threshold = 0.

        self.idx = torch.zeros(num_memory).long()
        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
        self.register_buffer("mIoU", torch.zeros(num_memory, 3).float())

        if self.use_part_feat:
            self.register_buffer("bottom_features", torch.zeros(num_memory, num_features))
            self.register_buffer("top_features", torch.zeros(num_memory, num_features))
        if self.co_learning:
            self.register_buffer("label2s", torch.zeros(num_memory).long())
    
    @torch.no_grad()
    def _init_cluster(self, cluster_mean, cluster_std):
        self.register_buffer("cluster_mean", torch.zeros_like(cluster_mean))
        self.register_buffer("cluster_std", torch.zeros_like(cluster_std))
        self.cluster_mean.data.copy_(cluster_mean.float().to(self.labels.device))
        self.cluster_std.data.copy_(cluster_std.float().to(self.labels.device))

    def _del_cluster(self):
        delattr(self, 'cluster_mean')
        delattr(self, 'cluster_std')
    
    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_bottom_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.bottom_features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_top_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.top_features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))
        self.mIoU.data.copy_(torch.ones(len(self.mIoU), 3).float()).to(self.mIoU.device)
    
    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    @torch.no_grad()
    def get_features(self, indexes):
        return self.features[indexes].clone()

    @torch.no_grad()
    def get_all_cluster_ids(self):
        return self.labels.clone()

    @torch.no_grad()
    def get_all_features(self):
        return self.features.clone()

    def masked_softmax_multi_focal(self, vec, mask, dim=1, targets=None, epsilon=1e-6, IoU=None, indexes=None, labels=None):
        """
            :vec: [B, u], 与聚类中心的相似度
            :mask: [B, u], 某些簇的数量为0
            :targets: [B, 1]
            :labels: [N, ]
        """
        exps = torch.exp(vec)   # [B, u]
        one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=exps.shape[1])
        one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)  # 返回一个与size大小相同的用1填充的张量
        one_hot_neg = one_hot_neg - one_hot_pos
        masked_exps = exps * mask.float().clone()
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

    def get_hard_cluster_loss_cluster(self, labels, sim, targets, IoU, indexes):
        """
            :sim: [B, C]
            :targets: [B]
            :labels: [N]
            :IoU: [N]
        """
        B = sim.shape[0]
        self.num_memory = labels.shape[0]
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) # [C], 求每一个簇样本个数
        mask = (nums > 0).float()
        sim = sim.t()

        mask = mask.expand_as(sim)
        masked_sim = self.masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets, IoU=IoU, indexes=indexes, labels=labels) # sim: u * B, mask:u * B, masked_sim: B * u
        cluster_hard_loss = F.nll_loss(torch.log(masked_sim + 1e-6), targets, reduce=False)
        cluster_hard_loss = cluster_hard_loss.mean()
        return cluster_hard_loss#, average_std, average_std_exclude_outliers

    def get_hard_cluster_loss(self, labels, cluster_inputs, targets, IoU, indexes, features):
        """
            :cluster_inputs: [B, N]
            :targets: [B]
            :labels: [N]
            :IoU: [N]
        """
        B = cluster_inputs.shape[0]
        self.num_memory = labels.shape[0]
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) # [C], 求每一个簇样本个数
        mask = (nums > 0).float()
        # 先求样本之间的相似度,再求和聚类中心的相似度
        sim = torch.zeros(labels.max() + 1, B).float().cuda() # C * B, unique label num: C = labels.max() + 1表示标签的数量
        sim.index_add_(0, labels, cluster_inputs.t().contiguous())  # 每一列表示minibatch中instance与同一个簇中所有instance的相似度的和
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) # average features in each cluster, C * B, 与聚类中心的相似度
        
        mask = mask.expand_as(sim)
        masked_sim = self.masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets, IoU=IoU, indexes=indexes, labels=labels) # sim: u * B, mask:u * B, masked_sim: B * u
        cluster_hard_loss = F.nll_loss(torch.log(masked_sim + 1e-6), targets, reduce=False)
        cluster_hard_loss = cluster_hard_loss.mean()
        return cluster_hard_loss

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
        cluster_center /= (mask * nums + (1 - mask)).clone().expand_as(cluster_center) # average features in each cluster, C * B, 与聚类中心的相似度
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

    def forward(self, feats, indexes, IoU, part_feats, top_IoU, bottom_IoU, pos_is_gt_list):
        """
            :feats: [B, 256]
            :indexes: [B, ]
            :IoU: [B, ]
        """
        
        update_flag, iou_target = self.get_update_flag(indexes, IoU)
        top_update_flag, top_iou_target = self.get_update_flag(indexes, top_IoU)
        bottom_update_flag, bottom_iou_target = self.get_update_flag(indexes, bottom_IoU)

        losses = {}
        targets = self.labels[indexes].clone()
        labels = self.labels.clone() # [N, ]
            
        feats = F.normalize(feats, p=2, dim=1)
        if self.use_part_feat:
            bottom_feats = F.normalize(part_feats[:, :256], p=2, dim=1)
            top_feats = F.normalize(part_feats[:, 256:], p=2, dim=1)
            inputs, bottom_inputs, top_inputs = hm_part(feats, bottom_feats, top_feats, indexes, labels, self.cluster_mean, self.cluster_std, self.features, self.bottom_features, \
                                                self.top_features, self.mIoU, self.momentum, self.IoU_momentum, IoU, top_IoU, bottom_IoU, self.IoU_memory_clip, \
                                                update_flag, top_update_flag, bottom_update_flag, self.update_method, targets)   # [B, N]
            inputs, bottom_inputs, top_inputs = inputs / self.temp, bottom_inputs / self.temp, top_inputs / self.temp
        else:
            inputs = hm(feats, indexes, self.features, IoU, self.update_method, self.momentum, update_flag, self.IoU_memory_clip)   # [B, N]
            inputs /= self.temp

        # losses["m2o_loss"] = self.get_m2o_loss(feats, targets, pos_is_gt_list)
        if self.use_max_IoU_bbox:
            inputs, global_targets, IoU, global_indexes, feats = inputs[update_flag], targets[update_flag], \
                            IoU[update_flag], indexes[update_flag], feats[update_flag]
            if self.use_part_feat:
                top_inputs, top_targets, top_IoU, top_indexes, top_feats = top_inputs[top_update_flag], targets[top_update_flag], \
                                top_IoU[top_update_flag], indexes[top_update_flag], top_feats[top_update_flag]
                bottom_inputs, bottom_targets, bottom_IoU, bottom_indexes, bottom_feats = bottom_inputs[bottom_update_flag], \
                            targets[bottom_update_flag], bottom_IoU[bottom_update_flag], indexes[bottom_update_flag], bottom_feats[bottom_update_flag]
        else:
            global_targets, bottom_targets, top_targets = targets.clone(), targets.clone(), targets.clone()
            global_indexes, bottom_indexes, top_indexes = indexes.clone(), indexes.clone(), indexes.clone()
        
        if self.use_cluster_hard_loss:
            losses["global_cluster_hard_loss"] = torch.tensor(0.)
            losses["part_cluster_hard_loss"] = torch.tensor(0.)
            if targets.shape[0] > 0:
                losses["global_cluster_hard_loss"]= self.get_hard_cluster_loss_cluster(labels.clone(), inputs, global_targets, IoU, global_indexes)
                # losses["average_std",], losses["average_std_exclude_outliers"] 
                # losses["global_iou_loss"] = self.get_iou_loss(feats, iou_target)
                if self.use_part_feat:
                    bottom_cluster_hard_loss = self.get_hard_cluster_loss(labels.clone(), bottom_inputs, bottom_targets, bottom_IoU, bottom_indexes, bottom_feats)
                    top_cluster_hard_loss = self.get_hard_cluster_loss(labels.clone(), top_inputs, top_targets, top_IoU, top_indexes, top_feats)
                    losses["part_cluster_hard_loss"] = bottom_cluster_hard_loss + top_cluster_hard_loss
                    # losses["part_iou_loss"] = self.get_iou_loss(bottom_feats, top_iou_target) + self.get_iou_loss(top_feats, bottom_iou_target)
        return losses