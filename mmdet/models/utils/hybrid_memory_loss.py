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
import min_search_cuda
import max_search_cuda
from torch.cuda.amp import custom_fwd, custom_bwd

class HM_part(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, bottom_inputs, top_inputs, indexes, features, bottom_features, top_features, mIoU, \
                    IoU, top_IoU, bottom_IoU, momentum, IoU_momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.mIoU = mIoU
        ctx.IoU_momentum = IoU_momentum
        ctx.bottom_features = bottom_features
        ctx.top_features = top_features

        outputs = inputs.mm(ctx.features.t())
        bottom_outputs = bottom_inputs.mm(ctx.bottom_features.t())
        top_outputs = top_inputs.mm(ctx.top_features.t())

        all_inputs = all_gather_tensor(inputs)
        all_indexes = all_gather_tensor(indexes)
        all_IoU = all_gather_tensor(IoU)
        all_top_IoU = all_gather_tensor(top_IoU)
        all_bottom_IoU = all_gather_tensor(bottom_IoU)
        bottom_inputs = all_gather_tensor(bottom_inputs)
        top_inputs = all_gather_tensor(top_inputs)

        ctx.save_for_backward(all_inputs, all_indexes, all_IoU, all_top_IoU, all_bottom_IoU, bottom_inputs, top_inputs)
        return outputs, bottom_outputs, top_outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs, grad_bottom_outputs, grad_top_outputs):
        inputs, indexes, IoU, top_IoU, bottom_IoU, bottom_inputs, top_inputs = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
            grad_bottom_outputs = grad_bottom_outputs.mm(ctx.bottom_features)
            grad_top_outputs = grad_top_outputs.mm(ctx.top_features)

        for x, y, iou, b, t, biou, tiou in zip(inputs, indexes, IoU, bottom_inputs, top_inputs, bottom_IoU, top_IoU):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
            
            # ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * (b if biou > tiou else t)
            # ctx.features[y] /= ctx.features[y].norm()

            # ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * b
            # ctx.features[y] /= ctx.features[y].norm()

            # ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * t
            # ctx.features[y] /= ctx.features[y].norm()


            ctx.mIoU[y] = ctx.IoU_momentum * ctx.mIoU[y] + (1.0 - ctx.IoU_momentum) * iou
            ctx.bottom_features[y] = ctx.momentum * ctx.bottom_features[y] + (1.0 - ctx.momentum) * b
            ctx.bottom_features[y] /= ctx.bottom_features[y].norm()
            ctx.top_features[y] = ctx.momentum * ctx.top_features[y] + (1.0 - ctx.momentum) * t
            ctx.top_features[y] /= ctx.top_features[y].norm()


        return grad_inputs, grad_bottom_outputs, grad_top_outputs, None, None, None, None, None, None, None, None, None, None

def hm_part(inputs, bottom_inputs, top_inputs, indexes, features, bottom_features, top_features, mIoU, momentum, IoU_momentum, IoU, top_IoU, bottom_IoU):
    return HM_part.apply(
        inputs, bottom_inputs, top_inputs, indexes, features, bottom_features, top_features, mIoU, IoU, top_IoU, bottom_IoU, \
                            torch.Tensor([momentum]).to(inputs.device), torch.Tensor([IoU_momentum]).to(inputs.device)
    )

class HM(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, indexes, features, mIoU, IoU, momentum, IoU_momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.mIoU = mIoU
        ctx.IoU_momentum = IoU_momentum

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
        for x, y, z in zip(inputs, indexes, IoU):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
            ctx.mIoU[y] = ctx.IoU_momentum * ctx.mIoU[y] + (1.0 - ctx.IoU_momentum) * z

        return grad_inputs, None, None, None, None, None, None

def hm(inputs, indexes, features, mIoU, IoU, momentum=0.5, IoU_momentum=0.2):
    return HM.apply(
        inputs, indexes, features, mIoU, IoU, torch.Tensor([momentum]).to(inputs.device), torch.Tensor([IoU_momentum]).to(inputs.device)
    
    )

class HybridMemoryMultiFocalPercent(nn.Module):

    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, cluster_top_percent=0.1, instance_top_percent=1, \
                    use_cluster_hard_loss=True, use_instance_hard_loss=False, use_hybrid_loss=False, testing=False, use_uncertainty_loss=False,
                    use_IoU_loss=False, use_IoU_memory=False, IoU_loss_clip=[0.7, 1.0], IoU_memory_clip=[0.7, 1.0], IoU_momentum=0.2,
                    foreground_weight=0.9, use_part_feat=False, co_learning=False, use_hard_mining=False):
        super(HybridMemoryMultiFocalPercent, self).__init__()
        self.use_cluster_hard_loss = use_cluster_hard_loss
        self.use_instance_hard_loss = use_instance_hard_loss
        self.use_hybrid_loss = use_hybrid_loss
        self.use_mini_instance_hard_loss = False
        self.num_features = num_features
        self.use_IoU_loss = use_IoU_loss
        self.use_IoU_memory = use_IoU_memory
        self.IoU_loss_clip = IoU_loss_clip
        self.IoU_memory_clip = IoU_memory_clip
        self.IoU_momentum = IoU_momentum
        self.use_gt = False
        self.foreground_weight = foreground_weight
        self.use_part_feat = use_part_feat

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

        self.idx = torch.zeros(num_memory).long()
        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
        self.register_buffer("mIoU", torch.zeros(num_memory).float())

        if self.use_part_feat:
            self.register_buffer("bottom_features", torch.zeros(num_memory, num_features))
            self.register_buffer("top_features", torch.zeros(num_memory, num_features))
        if self.co_learning:
            self.register_buffer("label2s", torch.zeros(num_memory).long())
    
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
        self.mIoU.data.copy_(torch.ones(len(self.mIoU)).float()).to(self.mIoU.device)

    @torch.no_grad()
    def _update_label2(self, labels):
        self.label2s.data.copy_(labels.long().to(self.label2s.device))
    
    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    @torch.no_grad()
    def get_cluster_id2s(self, indexes):
        return self.label2s[indexes].clone()
    
    def load_uncertainty(self, ):
        # path = './uncertainty'
        # files = os.listdir(path)
        # number = [int(file.split('.')[0].split('_')[-1]) for file in files]
        # sorted_num = sorted(number)
        # return os.path.join(path, 'uncertainty_{}.pth'.format(sorted_num[-1]))
        return  os.path.join('saved_file', 'uncertainty.pth')

    def masked_softmax_multi_focal(self, vec, mask, dim=1, targets=None, epsilon=1e-6, IoU=None, indexes=None, labels=None):
        """
            :vec: [B, u], 与聚类中心的相似度
            :mask: [B, u], 某些簇的数量为0
            :targets: [B, 1]
            :labels: [N, ]
        """
        exps = torch.exp(vec)   # [B, u]
        # if self.use_IoU_loss:
        #     clip_IoU = torch.clamp(IoU, self.IoU_loss_clip[0], self.IoU_loss_clip[1])    # [B, ]
        #     exps = exps * clip_IoU.detach().unsqueeze(1)

        # if self.use_uncertainty_loss:
        #     # 求簇的平均不确定性
        #     files_path = self.load_uncertainty()
        #     uncertainty = torch.load(files_path).cuda()
        #     cluster_uncertainty = torch.zeros(labels.max() + 1, 1).float().cuda() 
        #     cluster_uncertainty.index_add_(0, labels, uncertainty)
        #     nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        #     nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda())
        #     mask_ = (nums > 0).float()
        #     cluster_uncertainty /= (mask_ * nums + (1 - mask_)).clone().expand_as(cluster_uncertainty) # average features in each cluster, u * B, 与聚类中心的相似度
        #     cluster_uncertainty = cluster_uncertainty.squeeze(-1)

        #     instance_uncertainty = uncertainty[indexes] # [B]
        #     # weight = instance_uncertainty.unsqueeze(1)
        #     # weight = instance_uncertainty.unsqueeze(1) + cluster_uncertainty.unsqueeze(0)
        #     weight = cluster_uncertainty.unsqueeze(0)
        #     exps = exps * weight
        
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

    def get_all_hard_instance_loss(self, inputs, labels, targets, epsilon=1e-6):
        B = inputs.size(0)
        num_classes = labels.max() + 1
        cluster_idxs = []
        for lb in range(num_classes):
            c_idx = torch.nonzero((labels == lb)).view(-1).tolist()
            cluster_idxs.append(c_idx)
        min_idx = min_search_cuda.min_negative_search(inputs.cpu().detach().numpy(), cluster_idxs)
        max_idx = max_search_cuda.max_positive_search(inputs.cpu().detach().numpy(), targets.tolist(), cluster_idxs)

        one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)  # 返回一个与size大小相同的用1填充的张量
        one_hot_neg = one_hot_neg - one_hot_pos

        sim = torch.zeros((B, num_classes)).cuda()
        min_idx1 = torch.arange(min_idx.shape[0]).repeat(min_idx.shape[1], 1).transpose(0, 1).contiguous().view(-1)
        min_idx2 = torch.tensor(min_idx).view(-1).long()
        sim = inputs[min_idx1, min_idx2].reshape(B, num_classes)
        sim[one_hot_neg > 0] = inputs[min_idx1, min_idx2].reshape(B, num_classes)[one_hot_neg > 0]

        max_idx1 = torch.arange(max_idx.shape[0]).repeat(max_idx.shape[1], 1).transpose(0, 1).contiguous().view(-1)
        max_idx2 = torch.tensor(max_idx).view(-1).long()
        sim[one_hot_pos > 0] = inputs[max_idx1, max_idx2].reshape(B)
    
        sim_exp = torch.exp(sim).clone()
        neg_exps = sim_exp.new_zeros(size=sim_exp.shape)
        neg_exps[one_hot_neg > 0] = sim_exp[one_hot_neg > 0]

        # hardmining
        ori_neg_exps = neg_exps
        neg_exps = neg_exps / neg_exps.sum(dim=1, keepdim=True)

        sorted, indices = torch.sort(neg_exps, dim=1, descending=True)  # 排序得到相似度最大(难度最大)的负样本
        sorted_cum_sum = torch.cumsum(sorted, dim=1)
        sorted_cum_diff = (sorted_cum_sum - self.instance_top_percent).abs()
        sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)  # 获得K的大小
        min_values = sorted[torch.range(0, sorted.shape[0] - 1).long(), sorted_cum_min_indices]   # 获取K对应的val
        min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)   # 前面除neg_exps.sum(),所以这里乘回去
        ori_neg_exps[ori_neg_exps < min_values] = 0   # 小于阈值,即难度稍微小的负样本不考虑

        sim_exp[one_hot_neg > 0] = ori_neg_exps[one_hot_neg > 0]
        sim_sum = sim_exp.sum(dim=-1, keepdim=True) + epsilon
        sim_sum_softmax = sim_exp / sim_sum
        instance_hard_loss = F.nll_loss(torch.log(sim_sum_softmax + epsilon), targets)
        return instance_hard_loss

    def get_hybrid_loss(self, inputs, labels, targets, vec, mask, epsilon=1e-6):
        """
            以所在簇的聚类中心为正样本,其它簇的最难样本为负样本
        """

        exps = torch.exp(vec)
        cluster_exps = exps * mask.float().clone()

        B = inputs.size(0)
        num_classes = labels.max() + 1
        cluster_idxs = []
        for lb in range(num_classes):
            c_idx = torch.nonzero((labels == lb)).view(-1).tolist()
            cluster_idxs.append(c_idx)
        min_idx = min_search_cuda.min_negative_search(inputs.cpu().detach().numpy(), cluster_idxs)

        one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)  # 返回一个与size大小相同的用1填充的张量
        one_hot_neg = one_hot_neg - one_hot_pos

        sim = torch.zeros((B, num_classes)).cuda()
        min_idx1 = torch.arange(min_idx.shape[0]).repeat(min_idx.shape[1], 1).transpose(0, 1).contiguous().view(-1)
        min_idx2 = torch.tensor(min_idx).view(-1).long()
        sim[one_hot_neg > 0] = inputs[min_idx1, min_idx2].reshape(B, num_classes)[one_hot_neg > 0]

        sim_exp = torch.exp(sim).clone()
        sim_exp[one_hot_pos > 0] = cluster_exps[one_hot_pos > 0]

        neg_exps = sim_exp.new_zeros(size=sim_exp.shape)
        neg_exps[one_hot_neg > 0] = sim_exp[one_hot_neg > 0]

        # hardmining
        ori_neg_exps = neg_exps
        neg_exps = neg_exps / neg_exps.sum(dim=1, keepdim=True)

        sorted, indices = torch.sort(neg_exps, dim=1, descending=True)  # 排序得到相似度最大(难度最大)的负样本
        sorted_cum_sum = torch.cumsum(sorted, dim=1)
        sorted_cum_diff = (sorted_cum_sum - self.instance_top_percent).abs()
        sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)  # 获得K的大小
        min_values = sorted[torch.range(0, sorted.shape[0] - 1).long(), sorted_cum_min_indices]   # 获取K对应的val
        min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)   # 前面除neg_exps.sum(),所以这里乘回去
        ori_neg_exps[ori_neg_exps < min_values] = 0   # 小于阈值,即难度稍微小的负样本不考虑

        sim_exp[one_hot_neg > 0] = ori_neg_exps[one_hot_neg > 0]
        sim_sum = sim_exp.sum(dim=-1, keepdim=True) + epsilon
        sim_sum_softmax = sim_exp / sim_sum
        hybrid_loss = F.nll_loss(torch.log(sim_sum_softmax + epsilon), targets)
        return hybrid_loss

    def get_hard_cluster_loss(self, labels, cluster_inputs, targets, IoU, indexes):
        """
            :cluster_inputs: [B, N]
            :targets: [B]
            :labels: [N]
        """
        B = cluster_inputs.shape[0]

        # if self.hard_mining:    # 聚类中心不能考虑难样本
        #     files_path = self.load_uncertainty()
        #     uncertainty = torch.load(files_path).cuda()
        #     cluster_inputs = cluster_inputs * uncertainty.unsqueeze(0)
        #     labels[uncertainty == 0] = labels.max() + 1

        #     sim_ = torch.zeros(labels.max() + 1, B).float().cuda() # C * B, unique label num: C = labels.max() + 1表示标签的数量
        #     sim_.index_add_(0, labels, cluster_inputs.t().contiguous())  # 每一列表示minibatch中instance与同一个簇中所有instance的相似度的和
        #     nums_ = torch.zeros(labels.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
        #     nums_.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) # C * 1
        #     sim = sim_[:-1].clone()
        #     nums = nums_[:-1].clone()
        # else:
        sim = torch.zeros(labels.max() + 1, B).float().cuda() # C * B, unique label num: C = labels.max() + 1表示标签的数量
        sim.index_add_(0, labels, cluster_inputs.t().contiguous())  # 每一列表示minibatch中instance与同一个簇中所有instance的相似度的和
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
        self.num_memory = labels.shape[0]
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) # C * 1
        
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) # average features in each cluster, C * B, 与聚类中心的相似度
        mask = mask.expand_as(sim)
        masked_sim = self.masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets, IoU=IoU, indexes=indexes, labels=labels) # sim: u * B, mask:u * B, masked_sim: B * u
        cluster_hard_loss = F.nll_loss(torch.log(masked_sim + 1e-6), targets, reduce=False)
        
        # label_mask = torch.load(os.path.join('saved_file', 'label_mask.pth')).cuda()
        # target_label_mask = label_mask[indexes]
        # cluster_hard_loss = cluster_hard_loss * target_label_mask
        cluster_hard_loss = cluster_hard_loss.mean()
        
        return cluster_hard_loss

    def get_hard_cluster_loss_by_two_pseudo_labels(self, label1s, label2s, cluster_inputs, targets, target2s, IoU, indexes, features):
        """
            cluster_inputs: [B, N]
        """
        B = cluster_inputs.shape[0]
        # label1s = label1s.unsqueeze(0).repeat(B, 1)  # [B, N]
        # pseudo1_pos = (label1s == targets.unsqueeze(-1))    # labels: [B, N], targets:[B, 1], pseudo1_pos: [B, N]
        # label2s = label2s.unsqueeze(0).repeat(B, 1)  # [B, N]
        # pseudo2_pos = (label2s == target2s.unsqueeze(-1))    # labels: [B, N], targets:[B, 1], pseudo1_pos: [B, N]
        # complementary_pseudo_pos = pseudo1_pos ^ pseudo2_pos    # 补集
        # # 修改正样本簇补集元素为labels.max() + 1, 对比学习不考虑这些样本
        # label1s[complementary_pseudo_pos == True] = label1s.max() + 1

        sim_ = torch.zeros(label1s.max() + 1, B).float().cuda() # C * B, unique label num: C = labels.max() + 1表示标签的数量
        sim_.index_add_(0, label1s, cluster_inputs.t().contiguous())  # 每一列表示minibatch中instance与同一个簇中所有instance的相似度的和
        nums_ = torch.zeros(label1s.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
        nums_.index_add_(0, label1s, torch.ones(self.num_memory, 1).float().cuda()) # C * 1
        sim = sim_[:-1].clone()
        nums = nums_[:-1].clone()

        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) # average features in each cluster, C * B, 与聚类中心的相似度
        mask = mask.expand_as(sim)
        masked_sim = self.masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets, IoU=IoU, indexes=indexes, labels=label1s) # sim: u * B, mask:u * B, masked_sim: B * u
        cluster_hard_loss = F.nll_loss(torch.log(masked_sim + 1e-6), targets, reduce=False)

        # features: [N, 256]
        cluster_center = torch.zeros(label1s.max() + 1, features.shape[-1]).float().cuda()  # [C, 256]
        cluster_center.index_add_(0, label1s, features.contiguous())  # 每一列表示minibatch中instance与同一个簇中所有instance的相似度的和
        nums_ = torch.zeros(label1s.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
        nums_.index_add_(0, label1s, torch.ones(self.num_memory, 1).float().cuda()) # C * 1
        mask = (nums > 0).float()
        cluster_center /= (mask * nums + (1 - mask)).clone().expand_as(cluster_center) # average features in each cluster, C * B, 与聚类中心的相似度

        cluster_center2 = torch.zeros(label2s.max() + 1, features.shape[-1]).float().cuda()  # [C, 256]
        cluster_center2.index_add_(0, label2s, features.contiguous())  # 每一列表示minibatch中instance与同一个簇中所有instance的相似度的和
        mask = (nums > 0).float()
        cluster_center2 /= (mask * nums + (1 - mask)).clone().expand_as(cluster_center2) # average features in each cluster, C * B, 与聚类中心的相似度

        cluster_hard_loss = cluster_hard_loss.mean()
        return cluster_hard_loss

    def forward(self, feats, indexes, IoU, part_feats, top_IoU, bottom_IoU):
        """
            :inputs: [B, 256]
            :indexes: [B, ]
            :IoU: [B, ]
            :cls_score_pos: [B, ]
            :bbox_targets: [B, 256]
        """
        inputs = F.normalize(feats, p=2, dim=1)
        if self.use_part_feat:
            bottom_inputs = F.normalize(part_feats[:, :256], p=2, dim=1)
            top_inputs = F.normalize(part_feats[:, 256:], p=2, dim=1)
            inputs, bottom_inputs, top_inputs = hm_part(inputs, bottom_inputs, top_inputs, indexes, self.features, self.bottom_features, \
                                                self.top_features, self.mIoU, self.momentum, self.IoU_momentum, IoU, top_IoU, bottom_IoU)   # [B, N]
            inputs, bottom_inputs, top_inputs = inputs / self.temp, bottom_inputs / self.temp, top_inputs / self.temp
        else:
            inputs = hm(inputs, indexes, self.features, self.mIoU, IoU, self.momentum, self.IoU_momentum)   # [B, N]
            inputs /= self.temp

        losses = {}
        targets = self.labels[indexes].clone()
        labels = self.labels.clone() # [N, ]
    
        if self.co_learning:
            target2s = self.label2s[indexes].clone()
            label2s = self.label2s.clone() # [N, ]

        # cluster_inputs = inputs # [B, N]
        # if self.use_IoU_memory:
        #     input_t = inputs.t()    # [N, B]
        #     IoU_weight = self.foreground_weight * self.mIoU + (1 - self.foreground_weight) * (1 - self.mIoU)
        #     # clip_IoU = torch.clamp(self.mIoU.detach(), self.IoU_memory_clip[0], self.IoU_memory_clip[1])    # [N, ]
        #     cluster_inputs = input_t * IoU_weight.unsqueeze(1)    # [N, B] * [N, ] = [N, B]
        #     cluster_inputs = cluster_inputs.t() # [B, N]

        label_mask = torch.load(os.path.join('saved_file', 'label_mask.pth')).cuda()
        target_label_mask = label_mask[indexes]
        labels = labels[label_mask]
        inputs = inputs[target_label_mask]
        inputs = inputs.t()[label_mask].t()
        bottom_inputs = bottom_inputs[target_label_mask]
        bottom_inputs = bottom_inputs.t()[label_mask].t()
        top_inputs = top_inputs[target_label_mask]
        top_inputs = top_inputs.t()[label_mask].t()
        targets = targets[target_label_mask]
        
        if self.use_cluster_hard_loss:
            losses["global_cluster_hard_loss"] = torch.tensor(0.)
            losses["part_cluster_hard_loss"] = torch.tensor(0.)
            if targets.shape[0] > 0:
                losses["global_cluster_hard_loss"] = self.get_hard_cluster_loss(labels.clone(), inputs, targets, IoU, indexes)
                if self.use_part_feat:
                    bottom_cluster_hard_loss = self.get_hard_cluster_loss(labels.clone(), bottom_inputs, targets, bottom_IoU, indexes)
                    top_cluster_hard_loss = self.get_hard_cluster_loss(labels.clone(), top_inputs, targets, top_IoU, indexes)
                    losses["part_cluster_hard_loss"] = bottom_cluster_hard_loss + top_cluster_hard_loss
            
            if self.co_learning:
                losses["global_cluster_hard_loss2"] = self.get_hard_cluster_loss(label2s.clone(), inputs, target2s, IoU, indexes)
                if self.use_part_feat:
                    bottom_cluster_hard_loss = self.get_hard_cluster_loss(label2s.clone(), bottom_inputs, target2s, bottom_IoU, indexes)
                    top_cluster_hard_loss = self.get_hard_cluster_loss(label2s.clone(), top_inputs, target2s, top_IoU, indexes)
                    losses["part_cluster_hard_loss2"] = bottom_cluster_hard_loss + top_cluster_hard_loss
                else:
                    losses["part_cluster_hard_loss2"] = torch.tensor(0.)
            
            # if self.co_learning:
            #     losses["global_cluster_hard_loss"] = self.get_hard_cluster_loss_by_two_pseudo_labels(labels, label2s, inputs, targets, target2s, IoU, indexes, self.features)
            #     if self.use_part_feat:
            #         bottom_cluster_hard_loss = self.get_hard_cluster_loss_by_two_pseudo_labels(labels, label2s, bottom_inputs, targets, target2s, bottom_IoU, indexes, self.bottom_features)
            #         top_cluster_hard_loss = self.get_hard_cluster_loss_by_two_pseudo_labels(labels, label2s, top_inputs, targets, target2s, top_IoU, indexes, self.top_features)
            #         losses["part_cluster_hard_loss"] = bottom_cluster_hard_loss + top_cluster_hard_loss
            #     else:
            #         losses["part_cluster_hard_loss"] = torch.tensor(0.)

        if self.use_instance_hard_loss:
            losses["instance_hard_loss"] = self.get_all_hard_instance_loss(inputs, labels, targets)

        if self.use_hybrid_loss:
            B = inputs.size(0)
            targets = self.labels[indexes].clone()
            labels = self.labels.clone() # shape: N
            sim = torch.zeros(labels.max() + 1, B).float().cuda() # u * B, unique label num: u = labels.max() + 1表示标签的数量
            sim.index_add_(0, labels, inputs.t().contiguous())  # 每一列表示minibatch中instance与同一个簇中所有instance的相似度的和
            nums = torch.zeros(labels.max() + 1, 1).float().cuda() # many instances belong to a cluster, so calculate the number of instances in a cluster
            nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) # u * 1
            mask = (nums > 0).float()
            sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) # average features in each cluster, u * B, 与聚类中心的相似度
            mask = mask.expand_as(sim)
            hybrid_loss = self.get_hybrid_loss(inputs, labels, targets, sim.t().contiguous(), mask.t().contiguous()) # sim: u * B, mask:u * B, masked_sim: B * u
            losses["hybrid_loss"] = hybrid_loss

        return losses
