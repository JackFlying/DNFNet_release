from tkinter import dialog
from typing import IO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms, multiclass_nms_aug)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import  Quaduplet2Loss, MemoryQuaduplet2Loss, HybridMemoryMultiFocalPercentCluster, HybridMemoryMultiFocalPercentDnfnet, HybridMemoryMultiFocalPercentClusterUnlabeled
from .gfn import GalleryFilterNetwork
from mmdet.models.utils.ProtoNorm import PrototypeNorm1d, register_targets_for_pn, convert_bn_to_pn
import os
from mmcv.ops import DeformConv2dPack
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from itertools import accumulate
import numpy as np


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels // 2)

        nn.init.normal_(self.fc1.weight, 0, 0.001)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.001)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c // 2, 1, 1)
        return y

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = np.power(input_dim, 0.5)

        self.fc_q = nn.Linear(input_dim, input_dim)
        self.fc_k = nn.Linear(input_dim, input_dim)
        self.fc_v = nn.Linear(input_dim, input_dim)

        self.fc_o = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value):
        # 将输入进行线性变换得到Q、K、V
        q = self.fc_q(query)  # [N, 1, D]
        k = self.fc_k(key)  # [N, 1, D]
        v = self.fc_v(value)  # [N, 1, D]

        # 将Q、K、V进行头分割
        q = self._split_heads(q)  # [N, num_heads, 1, head_dim]
        k = self._split_heads(k)  # [N, num_heads, 1, head_dim]
        v = self._split_heads(v)  # [N, num_heads, 1, head_dim]

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1))  # [N, num_heads, 1, 1]
        
        scores /= self.scale

        # 使用softmax进行归一化
        attention_weights = torch.softmax(scores, dim=-1)  # [N, num_heads, 1, 1]

        # 加权求和
        attended_value = torch.matmul(attention_weights, v)  # [N, num_heads, 1, head_dim]

        # 合并多头
        attended_value = self._combine_heads(attended_value)  # [N, 1, D]

        # 进行最终线性变换得到输出
        output = self.fc_o(attended_value)  # [N, 1, D]
        
        # 残差连接
        output = output + query

        return output

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.num_heads * self.head_dim)

@HEADS.register_module()
class CPCLHead(nn.Module):
    '''for person search, output reid features'''
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,    # 1
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_reid=dict(loss_weight=1.0),
                 rcnn_bbox_bn=False,
                 id_num=55272,
                 testing=False,
                 instance_top_percent=1.,
                 cluster_top_percent=0.1,
                 temperature=0.05,
                 momentum=0.2,
                 use_cluster_hard_loss=True,
                 cluster_mean_method='naive',
                 tc_winsize=500,
                 decay_weight=-0.0005,
                 tc_method='linear', 
                 intra_cluster_T=0.1,
                 use_part_feat=False,
                 use_quaduplet_loss=True,
                 use_max_IoU_bbox=False,
                 use_bn_affine=False,
                 update_method=None,
                 global_weight=0.9,
                 triplet_weight=1,
                 num_features=256,
                 margin=0.3,
                 triplet_bg_weight=0.25,
                 triplet_instance_weight=1,
                #  gfn_config=None,
                 norm_type='l2norm'):
        super(CPCLHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_reid = HybridMemoryMultiFocalPercentClusterUnlabeled(num_features, id_num, temperature, momentum, cluster_top_percent, instance_top_percent, use_cluster_hard_loss, testing, \
                                                        use_part_feat, use_max_IoU_bbox, update_method, cluster_mean_method, tc_winsize, intra_cluster_T, decay_weight, tc_method)
        
        self.loss_triplet = Quaduplet2Loss(margin=margin, bg_weight=triplet_bg_weight, instance_weight=triplet_instance_weight)
        self.use_quaduplet_loss = use_quaduplet_loss
        self.reid_loss_weight = loss_reid['loss_weight']
        self.triplet_weight = triplet_weight
        self.global_weight = global_weight
        # self.use_gfn = gfn_config['use_gfn']
        self.norm_type = norm_type   # ['l2norm', 'protonorm', 'batchnorm']
        self.use_bn_affine = use_bn_affine
        in_channels = self.in_channels
        self.deform_conv = DeformConv2dPack(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,)
    
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        
        self.rcnn_bbox_bn = rcnn_bbox_bn
        self.feat_channels = 512
        self.stacked_convs = 2
        reg_class_agnostic = False
        self.reid_feat_dim = 256
        
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='ReLU'),
                        bias='auto'),)
            if self.rcnn_bbox_bn:
                self.fc_reg = nn.Sequential(nn.Linear(self.feat_channels, out_dim_reg),
                                            nn.BatchNorm1d(out_dim_reg)
                )
            else:
                self.fc_reg = nn.Linear(self.feat_channels, out_dim_reg)

            self.cls_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='ReLU'), #None,#
                        bias='auto'),)
            self.fc_cls = nn.Linear(self.feat_channels, num_classes + 1)
            
        self.feature_h = 14
        self.feature_w = 6
        self.fc_reid = nn.Linear(in_channels * self.feature_h * self.feature_w, self.reid_feat_dim)
        # self.fc_part_reid = nn.ModuleList([nn.Linear(in_channels * self.feature_h * self.feature_w // 2, self.reid_feat_dim) for _ in range(2)])
        if self.norm_type is 'protonorm':
            self.normalize = PrototypeNorm1d(self.reid_feat_dim)
            self.normalize_part = nn.ModuleList([PrototypeNorm1d(self.reid_feat_dim), PrototypeNorm1d(self.reid_feat_dim)])
            self.bgnormalize = nn.BatchNorm1d(self.reid_feat_dim)
            self.bgnormalize_part = nn.ModuleList([nn.BatchNorm1d(self.reid_feat_dim), nn.BatchNorm1d(self.reid_feat_dim)]) 
        elif self.norm_type is 'batchnorm':
            self.normalize = nn.BatchNorm1d(num_features=self.reid_feat_dim, affine=self.use_bn_affine)
            self.normalize_part = nn.ModuleList([nn.BatchNorm1d(num_features=self.reid_feat_dim, affine=self.use_bn_affine), \
                                                nn.BatchNorm1d(num_features=self.reid_feat_dim, affine=self.use_bn_affine)])
            
            self.bgnormalize = nn.BatchNorm1d(num_features=self.reid_feat_dim, affine=self.use_bn_affine)
            self.bgnormalize_part = nn.ModuleList([nn.BatchNorm1d(num_features=self.reid_feat_dim, affine=self.use_bn_affine), \
                                    nn.BatchNorm1d(num_features=self.reid_feat_dim, affine=self.use_bn_affine)])

            self.normalize_std = nn.BatchNorm1d(num_features=self.reid_feat_dim, affine=self.use_bn_affine)
            self.bgnormalize_std = nn.BatchNorm1d(num_features=self.reid_feat_dim, affine=self.use_bn_affine)
        self.proposal_score_max = False
        # self.gt_fused_gru = nn.GRU(input_size=self.reid_feat_dim, hidden_size=self.reid_feat_dim, batch_first=True)
        
        # decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=2, dim_feedforward=1024, dropout=0.0, activation='relu')
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # self.se = SEBlock(in_channels * 2, reduction=16)
        # self.mha = MultiHeadCrossAttention(input_dim=self.reid_feat_dim, num_heads=4)
        
        # self.fc1 = nn.Linear(self.reid_feat_dim, self.reid_feat_dim)
        # self.fc2 = nn.Linear(self.reid_feat_dim, self.reid_feat_dim)

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
            for m in self.cls_convs:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
            for m in self.reg_convs:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
            
        nn.init.normal_(self.fc_reid.weight, 0, 0.001)
        nn.init.constant_(self.fc_reid.bias, 0)

        # for i in range(len(self.fc_part_reid)):
        #     nn.init.normal_(self.fc_part_reid[i].weight, 0, 0.001)
        #     nn.init.constant_(self.fc_part_reid[i].bias, 0)
        #     nn.init.normal_(self.fc_part_reid[i].weight, 0, 0.001)
        #     nn.init.constant_(self.fc_part_reid[i].bias, 0)

        # for name, param in self.gt_fused_gru.named_parameters():
        #     if 'weight' in name:
        #         nn.init.normal_(param, 0, 0.01)
    
        
    @auto_fp16()
    def crop_forward(self, crop_feats1, crop_feats2):
        crop_feat1 = crop_feats1.squeeze(-1).squeeze(-1)
        crop_feat2 = crop_feats2.squeeze(-1).squeeze(-1)
        crop_feat = F.normalize(torch.cat((self.id_feature(crop_feat2), self.id_feature1(crop_feat1)), axis=1))
        return crop_feat

    def get_iou(self, bbox_pred, rois, labels, bbox_targets):
        
        labels, label_weights, bbox_targets, bbox_weights, bbox_targets_xywh, pos_is_gt_list, crop_targets = bbox_targets
        
        labels = labels[:, 0]   # 0表示正样本(行人),  1表示负样本(背景)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes # bg_class_ind = 1
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1, 4)[pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)]]  # select pos sample

                decode_pos_bbox_pred = self.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_pred)
                pos_bbox_targets = bbox_targets_xywh[pos_inds.type(torch.bool)]

                top_pos_bbox_pred = decode_pos_bbox_pred.clone()   # [x1, y1, x2, y2]
                top_pos_bbox_pred[:, 1] = (top_pos_bbox_pred[:, 1] + top_pos_bbox_pred[:, 3]) / 2   # [x1, (y1+y2)/2, x2, y2]
                bottom_pos_bbox_pred = decode_pos_bbox_pred.clone()
                bottom_pos_bbox_pred[:, 3] = (bottom_pos_bbox_pred[:, 1] + bottom_pos_bbox_pred[:, 3]) / 2  # [x1, y1, x2, (y1+y2)/2]
                
                top_pos_bbox_targets = pos_bbox_targets.clone()
                top_pos_bbox_targets[:, 1] = (top_pos_bbox_targets[:, 1] + top_pos_bbox_targets[:, 3]) / 2
                bottom_pos_bbox_targets = pos_bbox_targets.clone()
                bottom_pos_bbox_targets[:, 3] = (bottom_pos_bbox_targets[:, 1] + bottom_pos_bbox_targets[:, 3]) / 2

                IoU = torchvision.ops.box_iou(decode_pos_bbox_pred, pos_bbox_targets)
                top_IoU = torchvision.ops.box_iou(top_pos_bbox_pred, top_pos_bbox_targets)
                bottom_IoU = torchvision.ops.box_iou(bottom_pos_bbox_pred, bottom_pos_bbox_targets)


                dialog = torch.eye(IoU.shape[0]).bool().cuda()
                IoU = IoU[dialog]
                top_IoU = top_IoU[dialog]
                bottom_IoU = bottom_IoU[dialog]

        return IoU, top_IoU, bottom_IoU

    def get_max_iou_pos(self, indexes, IoU):
        unique_labels = torch.unique(indexes)
        update_flag = torch.zeros_like(indexes).bool().to(indexes.device)
        for uid in unique_labels:
            IoU_tmp = IoU.clone()
            IoU_tmp[indexes!=uid] = -1
            maxid = torch.argmax(IoU_tmp)
            update_flag[maxid] = True
        return update_flag.long()

    def integrate_max_iou_context(self, feats, targets, indexes, pos_is_gt_list, IoU):
        """
            每张图片中的样本和gt proposal拉进
            pos_is_gt_list: gt的用1表示,预测的用0表示,但是预测和哪个gt之间有对应关系不明确
        """
        
        max_iou_pos = self.get_max_iou_pos(indexes, IoU)
        proposals_nums = [len(value) for value in pos_is_gt_list]
        pos_is_max_iou_list = max_iou_pos.split(proposals_nums)
        max_iou_nums = [torch.sum(value).item() for value in pos_is_max_iou_list]
        pred_nums = [proposals_nums[i] - max_iou_nums[i] for i in range(len(proposals_nums))]
        cumsum_pro_nums = list(accumulate([0] + proposals_nums))

        fused_pred_feats = []
        for i in range(1, len(cumsum_pro_nums)):
            max_iou_num = max_iou_nums[i-1]
            pred_num = pred_nums[i-1]
            
            gt_targets = targets[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_max_iou_list[i-1]==1]
            pred_targets = targets[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_max_iou_list[i-1]==0]
            
            gt_feats = feats[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_max_iou_list[i-1]==1]
            pred_feats = feats[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_max_iou_list[i-1]==0]

            if max_iou_num > 0:
                fused_pred_feats.append(gt_feats)   # 自身作为gt也要加入到list中
            if pred_num > 0:
                if max_iou_num > 0:
                    pred_gt_matrix = (gt_targets[:, None] == pred_targets[None]).int()
                    pred_gt_index = pred_gt_matrix.argmax(dim=0)
                    pred_gt_feats = gt_feats[pred_gt_index]

                    # 1. GRU融合, gt作为input
                    pred_gt_feats = pred_gt_feats[:, None, :]   # [N, L, D]
                    pred_feats = pred_feats[:, None, :].transpose(0, 1) # [N, D] -> [N, 1, D] -> [1, N, D]
                    output, hn = self.gt_fused_gru(pred_gt_feats, pred_feats)   # inputs, h0
                    update_pred_feats = hn.squeeze(0)
                    fused_pred_feats.append(update_pred_feats)
                    
                else:   # 有预测但是没有gt
                    fused_pred_feats.append(pred_feats)

        fused_pred_feats = torch.cat(fused_pred_feats, dim=0)
        assert fused_pred_feats.shape[0] == feats.shape[0]
        return fused_pred_feats

    def integrate_context_by_SE(self, feats, targets, pos_is_gt_list):
        """
            每张图片中的样本和gt proposal拉进
            pos_is_gt_list: gt的用1表示,预测的用0表示,但是预测和哪个gt之间有对应关系不明确
        """
        proposals_nums = [len(value) for value in pos_is_gt_list]
        gt_nums = [torch.sum(value).item() for value in pos_is_gt_list]
        pred_nums = [proposals_nums[i] - gt_nums[i] for i in range(len(proposals_nums))]
        cumsum_pro_nums = list(accumulate([0] + proposals_nums))

        fused_pred_feats = []
        for i in range(1, len(cumsum_pro_nums)):
            gt_num = gt_nums[i-1]
            pred_num = pred_nums[i-1]
            
            gt_targets = targets[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==1]
            pred_targets = targets[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==0]
            
            gt_feats = feats[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==1]
            pred_feats = feats[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==0]

            if gt_num > 0:
                fused_pred_feats.append(gt_feats)   # 自身作为gt也要加入到list中
            if pred_num > 0:
                if gt_num > 0:
                    pred_gt_matrix = (gt_targets[:, None] == pred_targets[None]).int()
                    pred_gt_index = pred_gt_matrix.argmax(dim=0)
                    pred_gt_feats = gt_feats[pred_gt_index]
                    # SE block融合
                    s = self.se(torch.cat([pred_gt_feats, pred_feats], dim=1))
                    updated_pred_feats = pred_feats + s * pred_gt_feats
                    fused_pred_feats.append(updated_pred_feats)
                else:   # 有预测但是没有gt
                    fused_pred_feats.append(pred_feats)
        # import ipdb;    ipdb.set_trace()
        fused_pred_feats = torch.cat(fused_pred_feats, dim=0)
        assert fused_pred_feats.shape[0] == feats.shape[0]
        return fused_pred_feats

    @auto_fp16()
    def forward(self, x, part_feats, labels, rois, bbox_targets, pos_is_gt_list):

        x = self.deform_conv(x)
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)  # ->[N, 512, 14, 6]
        cls_feat = F.adaptive_avg_pool2d(cls_feat, (1, 1)).view(cls_feat.size(0), -1)   # [N, 512]
        cls_score = self.fc_cls(cls_feat)   # [N, 2]

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        reg_feat = F.adaptive_avg_pool2d(reg_feat, (1, 1)).view(reg_feat.size(0), -1)
        bbox_pred = self.fc_reg(reg_feat)   # [N, 4]
        
        x_reid = x
        id_pred = self.fc_reid(x_reid.view(x_reid.size(0), -1))

        if self.training:
            id_labels = labels[:, 1]
        #     IoU, _, _ = self.get_iou(bbox_pred, rois, labels, bbox_targets)
        #     id_labels_fg = id_labels[id_labels != -2]
        #     id_pred_fg = id_pred[id_labels != -2]
        #     targets = self.loss_reid.get_cluster_ids(id_labels_fg)
        #     id_pred[id_labels != -2] = self.integrate_max_iou_context(id_pred_fg, targets, id_labels_fg, pos_is_gt_list, IoU)
        # else:
        #     # import ipdb;    ipdb.set_trace()
        #     IoU, _, _ = self.get_iou(bbox_pred, rois, labels, bbox_targets)
        #     id_pred[id_labels != -2] = self.integrate_max_iou_context(id_pred_fg, targets, id_labels_fg, pos_is_gt_list, IoU)
        
        id_pred = F.normalize(id_pred)
        if self.norm_type in ['protonorm'] and self.training:
            IoU, top_IoU, bottom_IoU = self.get_iou(bbox_pred, rois, labels, bbox_targets[2])
            person_id = id_labels.clone()
            person_id[id_labels!=-2] = self.loss_reid.get_cluster_ids(id_labels[id_labels!=-2])
            norm_flag = self.get_update_flag(id_labels[id_labels!=-2], IoU)
            bottom_update_flag = self.get_update_flag(id_labels[id_labels!=-2], bottom_IoU)
            top_update_flag = self.get_update_flag(id_labels[id_labels!=-2], top_IoU)
            register_targets_for_pn(self.normalize, person_id[id_labels!=-2].long(), norm_flag, IoU)
            register_targets_for_pn(self.normalize_part[0], person_id[id_labels!=-2].long(), bottom_update_flag, bottom_IoU)
            register_targets_for_pn(self.normalize_part[1], person_id[id_labels!=-2].long(), top_update_flag, top_IoU)

        if self.norm_type in ['protonorm', 'batchnorm']:
            if self.training:
                id_pred[id_labels!=-2] = self.normalize(id_pred[id_labels!=-2])
                id_pred[id_labels==-2] = self.bgnormalize(id_pred[id_labels==-2])
                # id_pred_log_var[id_labels!=-2] = self.normalize_std(id_pred_log_var[id_labels!=-2])
                # id_pred_log_var[id_labels==-2] = self.bgnormalize_std(id_pred_log_var[id_labels==-2])
            else:
                id_pred = self.normalize(id_pred)

        part_id_pred = None
        if part_feats is not None:
            part_id_pred = []
            for i in range(len(part_feats)):
                part_feat = self.deform_conv(part_feats[i].contiguous())
                id_feat = self.fc_part_reid[i](part_feat.view(part_feat.size(0), -1))
                if self.norm_type in ['protonorm', 'batchnorm']:
                    if self.training:
                        id_feat[id_labels!=-2] = self.normalize_part[i](id_feat[id_labels!=-2])
                        id_feat[id_labels==-2] = self.bgnormalize_part[i](id_feat[id_labels==-2])
                    else:
                        id_feat = self.normalize_part[i](id_feat)
                id_feat = F.normalize(id_feat)
                part_id_pred.append(id_feat)
            part_id_pred = torch.cat(part_id_pred, dim=1)   # [N, 512]
        return cls_score, bbox_pred, id_pred, part_id_pred

    def _get_target_single_crop(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, 
                           pos_gt_labels, pos_gt_crop_feats, cfg):
        """
            pos_gt_labels: [N, 4]
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg # 128
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, 3),
                                     self.num_classes,  # num_classes = 1
                                     dtype=torch.long)
        # background id is -2
        labels[:, 1] = -2
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels[:, :3]
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
            if pos_gt_crop_feats is not None:
                crop_targets = pos_bboxes.new_zeros(num_samples, 256)
                crop_targets[:num_pos, :] = pos_gt_crop_feats
            else:
                crop_targets = None

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, crop_targets

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, 
                           pos_gt_labels, cfg):
        """
            pos_gt_labels: [N, 4]
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg # 128
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, pos_gt_labels.shape[1]),
                                     self.num_classes,  # num_classes = 1
                                     dtype=torch.long)
        # background id is -2
        labels[:, 1] = -2
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_targets_xywh = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels[:, :]
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_targets_xywh[:num_pos, :] = pos_gt_bboxes
            bbox_weights[:num_pos, :] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, bbox_targets_xywh

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    **kwargs):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_is_gt_list = [res.pos_is_gt for res in sampling_results]

        if kwargs['use_crop']:
            pos_crop_feats_list = [res.pos_crop_feats for res in sampling_results]
            labels, label_weights, bbox_targets, bbox_weights, crop_targets = multi_apply(
                self._get_target_single_crop,
                pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                pos_crop_feats_list,
                cfg=rcnn_train_cfg)
        else:
            labels, label_weights, bbox_targets, bbox_weights, bbox_targets_xywh = multi_apply(
                self._get_target_single,
                pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg=rcnn_train_cfg)
            crop_targets = None

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_targets_xywh = torch.cat(bbox_targets_xywh, 0)
            if crop_targets is not None:
                crop_targets = torch.cat(crop_targets, 0)
        return labels, label_weights, bbox_targets, bbox_weights, bbox_targets_xywh, pos_is_gt_list, crop_targets

    def integrate_gt_context(self, feats, targets, pos_is_gt_list):
        """
            每张图片中的样本和gt proposal拉进
            pos_is_gt_list: gt的用1表示,预测的用0表示,但是预测和哪个gt之间有对应关系不明确
        """
        proposals_nums = [len(value) for value in pos_is_gt_list]
        gt_nums = [torch.sum(value).item() for value in pos_is_gt_list]
        pred_nums = [proposals_nums[i] - gt_nums[i] for i in range(len(proposals_nums))]
        cumsum_pro_nums = list(accumulate([0] + proposals_nums))

        fused_pred_feats = []
        for i in range(1, len(cumsum_pro_nums)):
            gt_num = gt_nums[i-1]
            pred_num = pred_nums[i-1]
            
            gt_targets = targets[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==1]
            pred_targets = targets[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==0]
            
            gt_feats = feats[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==1]
            pred_feats = feats[cumsum_pro_nums[i-1]:cumsum_pro_nums[i]][pos_is_gt_list[i-1]==0]

            if gt_num > 0:
                fused_pred_feats.append(gt_feats)   # 自身作为gt也要加入到list中
            if pred_num > 0:
                if gt_num > 0:
                    pred_gt_matrix = (gt_targets[:, None] == pred_targets[None]).int()
                    pred_gt_index = pred_gt_matrix.argmax(dim=0)
                    pred_gt_feats = gt_feats[pred_gt_index]

                    # 1. GRU融合, gt作为input
                    pred_gt_feats = pred_gt_feats[:, None, :]   # [N, L, D]
                    pred_feats = pred_feats[:, None, :].transpose(0, 1) # [N, D] -> [N, 1, D] -> [1, N, D]
                    output, hn = self.gt_fused_gru(pred_gt_feats, pred_feats)   # inputs, h0
                    update_pred_feats = hn.squeeze(0)

                    # 2. GRU融合, gt作为hidden state
                    # pred_gt_feats = pred_gt_feats[:, None, :].transpose(0, 1)
                    # pred_feats = pred_feats[:, None, :]
                    # output, hn = self.gt_fused_gru(pred_feats, pred_gt_feats)   # inputs, h0
                    # update_pred_feats = output.squeeze(1)
                    
                    # # 3. attention融合
                    # pred_gt_feats = pred_gt_feats[:, None, :]   # [N, 1, D] -> [1, N, D]
                    # pred_feats = pred_feats[:, None, :] # [N, 1, D] -> [1, N, D]
                    # update_pred_feats = self.mha(pred_feats, pred_gt_feats, pred_gt_feats)
                    # update_pred_feats = update_pred_feats.squeeze(1)
                    # # update_pred_feats = self.transformer_decoder(pred_gt_feats, pred_feats)   # tgt:解码器的序列; memory:编码器最后一层的序列;
                    # # update_pred_feats = update_pred_feats.squeeze(0)
                    fused_pred_feats.append(update_pred_feats)
                    
                else:   # 有预测但是没有gt
                    fused_pred_feats.append(pred_feats)

        fused_pred_feats = torch.cat(fused_pred_feats, dim=0)
        assert fused_pred_feats.shape[0] == feats.shape[0]
        return fused_pred_feats

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'id_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             id_pred,
             part_feats,
             id_pred_log_var,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             bbox_targets_xywh,
             pos_is_gt_list,
             crop_targets=None,
             reduction_override=None,
             **kwargs):
        id_labels = labels[:, 1]    # memory中的索引
        labels = labels[:, 0]   # 0表示正样本(行人),  1表示负样本(背景)
        losses = dict()
        
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                if cls_score.dim() == 2:
                    losses['loss_cls'] = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                    losses['acc'] = accuracy(cls_score, labels)
                elif cls_score.dim() == 1:
                    losses['loss_cls'] = F.binary_cross_entropy_with_logits(cls_score, labels.float())
                    cls_score_logit = F.sigmoid(cls_score).unsqueeze(-1)
                    cls_score = torch.cat([1 - cls_score_logit, cls_score_logit], dim=-1)
                    losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes # bg_class_ind = 1
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1, 4)[pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)]]  # select pos sample
                if crop_targets is not None:
                    pos_crop_targets = crop_targets[pos_inds.type(torch.bool)]
                else:
                    pos_crop_targets = None
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                
                cls_score_sf = F.softmax(cls_score, dim=-1)
                cls_score_sf = cls_score_sf[:, 0]

                decode_pos_bbox_pred = self.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_pred)
                pos_bbox_targets = bbox_targets_xywh[pos_inds.type(torch.bool)]
                
                top_pos_bbox_pred = decode_pos_bbox_pred.clone()   # [x1, y1, x2, y2]
                top_pos_bbox_pred[:, 1] = (top_pos_bbox_pred[:, 1] + top_pos_bbox_pred[:, 3]) / 2   # [x1, (y1+y2)/2, x2, y2]
                bottom_pos_bbox_pred = decode_pos_bbox_pred.clone()
                bottom_pos_bbox_pred[:, 3] = (bottom_pos_bbox_pred[:, 1] + bottom_pos_bbox_pred[:, 3]) / 2  # [x1, y1, x2, (y1+y2)/2]
                
                top_pos_bbox_targets = pos_bbox_targets.clone()
                top_pos_bbox_targets[:, 1] = (top_pos_bbox_targets[:, 1] + top_pos_bbox_targets[:, 3]) / 2
                bottom_pos_bbox_targets = pos_bbox_targets.clone()
                bottom_pos_bbox_targets[:, 3] = (bottom_pos_bbox_targets[:, 1] + bottom_pos_bbox_targets[:, 3]) / 2

                IoU = torchvision.ops.box_iou(decode_pos_bbox_pred, pos_bbox_targets)
                top_IoU = torchvision.ops.box_iou(top_pos_bbox_pred, top_pos_bbox_targets)
                bottom_IoU = torchvision.ops.box_iou(bottom_pos_bbox_pred, bottom_pos_bbox_targets)

                dialog = torch.eye(IoU.shape[0]).bool().cuda()
                IoU = IoU[dialog]
                top_IoU = top_IoU[dialog]
                bottom_IoU = bottom_IoU[dialog]
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                IoU = torch.zeros(0).cuda()
        
        rid_pred = id_pred[id_labels!=-2]   # [B, 256]
        rid_labels = id_labels[id_labels!=-2]
        rpart_feats = part_feats[id_labels!=-2] if part_feats is not None else None

        # targets = self.loss_reid.get_cluster_ids(rid_labels)
        # rid_pred = self.integrate_gt_context(rid_pred, targets, pos_is_gt_list)
        # id_pred[id_labels!=-2] = rid_pred
        
        memory_loss = self.loss_reid(rid_pred, rid_labels, IoU, top_IoU, bottom_IoU, rpart_feats)
        memory_loss['global_cluster_hard_loss'] *= self.global_weight
        # memory_loss["part_cluster_hard_loss"] *= (1 - self.global_weight)
        losses.update(memory_loss)

        if self.use_quaduplet_loss:
            cluster_id_labels = self.loss_reid.get_cluster_ids(id_labels[id_labels != -2])
            new_id_labels = id_labels.clone()
            new_id_labels[id_labels != -2] = cluster_id_labels
            losses['global_triplet_loss'] = self.loss_triplet(id_pred, new_id_labels, id_labels) * self.triplet_weight

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'id_pred', 'tmp_feat'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   id_pred,
                   tmp_feat,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        # 去掉测一下
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            if self.proposal_score_max:
                scores[:, 0] = 1
                scores[:, 1] = 0

            if tmp_feat is None:
                det_bboxes, det_labels, det_ids = multiclass_nms_aug(bboxes, scores, [id_pred],
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
                if det_ids is None:
                    det_ids = det_bboxes.new_zeros((0, 256))
                else:
                    det_ids = det_ids[0]
                det_bboxes = torch.cat([det_bboxes, det_ids], dim=1)

            else:
                det_bboxes, det_labels, det_ids = multiclass_nms_aug(bboxes, scores, [id_pred, tmp_feat],
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
                if det_ids is None:
                    det_ids, det_ids2 = det_bboxes.new_zeros((0, 256)), det_bboxes.new_zeros((0, tmp_feat.shape[-1]))
                else:
                    det_ids, det_ids2 = det_ids[0], det_ids[1]
                det_bboxes = torch.cat([det_bboxes, det_ids, det_ids2], dim=1)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
