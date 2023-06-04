from hashlib import new
import os.path as osp
from re import T
import warnings
from xml.etree.ElementTree import TreeBuilder

from mmcv.runner import Hook
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from mmdet.utils import get_dist_info
from mmdet.utils import all_gather_tensor, synchronize
from mmdet.core.label_generators import LabelGenerator
# from mmdet.models.utils import ClusterMemory
import collections
import mmcv
import os
import sys
from .utils import *


class ClusterHook(Hook):
    """Evaluation hook.

    Notes:
        If new arguments are added for EvalHook, tools/test.py may be
    effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation sta rting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, train_dataloaders, start=None, interval=1, logger=None, cfg=None, epoch_interval=1, **eval_kwargs):
        self.dataloaders = train_dataloaders
        self.datasets = [i.dataset for i in train_dataloaders]
        self.logger = logger
        self.cfg = cfg
        self.label_generator = LabelGenerator(self.cfg, self.dataloaders)
        self.epoch = 0
        self.epoch_interval = epoch_interval
        # self.uncertainty_estimation = False
        # self.co_learning = cfg.CO_LEARNING
        self.use_part_feat = cfg.PSEUDO_LABELS.part_feat.use_part_feat
        self.part_based_uncertainty = cfg.PSEUDO_LABELS.part_feat.uncertainty
        # self.use_k_reciprocal_nearest = cfg.PSEUDO_LABELS.use_k_reciprocal_nearest
        # self.co_learning = cfg.CO_LEARNING
    
    def before_train_epoch(self, runner):
        self.logger.info('start clustering for updating, pseudo labels')
        if self.epoch % self.epoch_interval != 0:
            self.epoch += 1
            return
        memory_features = []
        use_gaussion = True if hasattr(runner.model.module.roi_head.bbox_head.loss_reid, "features_std") is True else False

        memory_features_std = []
        start_ind = 0
        for idx, dataset in enumerate(self.datasets):
            if self.cfg.testing:
                dataset.id_num = 500

            memory_features.append(
                runner.model.module.roi_head.bbox_head.loss_reid
                .features[start_ind : start_ind + dataset.id_num]
                .clone()
                .cpu()
            )
            if use_gaussion:
                memory_features_std.append(
                    runner.model.module.roi_head.bbox_head.loss_reid
                    .features_std[start_ind : start_ind + dataset.id_num]
                    .clone()
                    .cpu() 
                )
            
            start_ind += dataset.id_num

            if not self.part_based_uncertainty:
                pseudo_labels, label_centers, pseudo_blabels, pseudo_tlabels = self.label_generator(
                    memory_features=memory_features, 
                    image_inds=runner.model.module.roi_head.bbox_head.loss_reid.idx[:len(memory_features[0])].clone().cpu(),
                    cfg=self.cfg
                )
                pseudo_labels = get_uncertainty_by_centroid(pseudo_labels, memory_features, self.logger, self.cfg.PSEUDO_LABELS.T)
                pseudo_labels = [pseudo_labels]

            else:
                self.cfg.PSEUDO_LABELS.part_feat.use_part_feat = True   # 为了防止新的epoch自动变成0
                pseudo_labels, label_centers, pseudo_blabels, pseudo_tlabels = self.label_generator(
                    memory_features=memory_features, 
                    image_inds=runner.model.module.roi_head.bbox_head.loss_reid.idx[:len(memory_features[0])].clone().cpu(),
                    cfg=self.cfg
                )
                if self.part_based_uncertainty:
                    self.cfg.PSEUDO_LABELS.part_feat.use_part_feat = False
                    pseudo_label2s, _, _, _ = self.label_generator(
                        memory_features=memory_features,
                        image_inds=runner.model.module.roi_head.bbox_head.loss_reid.idx[:len(memory_features[0])].clone().cpu(),
                        cfg=self.cfg
                    )
                    uncertainty = get_uncertainty_by_part(pseudo_labels, pseudo_label2s, memory_features, self.logger, self.cfg)
                    if self.cfg.PSEUDO_LABELS.hard_mining.use_hard_mining:
                        pseudo_labels = transfer_label_noise_to_outlier(uncertainty, pseudo_labels[0])
                    weight = get_weight_by_uncertainty(uncertainty, pseudo_labels[0])
                    torch.save(weight, os.path.join("saved_file", "weight.pth"))

            torch.save(pseudo_labels, os.path.join("saved_file", "pseudo_labels.pth"))

        # 统计每个类别的数量，将异常点单独存放在unlabel memory当中
        # pseudo_labels = instance_label(pseudo_labels)
        # pseudo_labels = get_outlier(pseudo_labels)

        memory_labels, memory_blabels, memory_tlabels = [], [], []
        start_pid = 0
        for idx, dataset in enumerate(self.datasets):
            labels = pseudo_labels[idx]
            memory_labels.append(torch.LongTensor(labels) + start_pid)
            if self.part_based_uncertainty and pseudo_blabels != []:
                blabels = pseudo_blabels[idx]
                tlabels = pseudo_tlabels[idx]
                memory_blabels.append(torch.LongTensor(blabels) + start_pid)
                memory_tlabels.append(torch.LongTensor(tlabels) + start_pid)
            else:
                blabels = pseudo_labels[idx]
                tlabels = pseudo_labels[idx]
                memory_blabels.append(torch.LongTensor(blabels) + start_pid)
                memory_tlabels.append(torch.LongTensor(tlabels) + start_pid)
                
            start_pid += max(labels) + 1
        memory_labels = torch.cat(memory_labels).view(-1)
        if self.part_based_uncertainty:
            memory_blabels = torch.cat(memory_blabels).view(-1)
            memory_tlabels = torch.cat(memory_tlabels).view(-1)
            
        runner.model.module.roi_head.bbox_head.loss_reid.clock = 0
        
        if hasattr(runner.model.module.roi_head.bbox_head.loss_reid, "use_cluster_memory"):
            if hasattr(runner.model.module.roi_head.bbox_head.loss_reid, "cluster_mean"):   # use mean
                runner.model.module.roi_head.bbox_head.loss_reid._del_cluster()
            if hasattr(runner.model.module.roi_head.bbox_head.loss_reid, "cluster_std"):    # feature uncertain, use mean and std
                means, stds = GMM(memory_features[0], memory_features_std[0], memory_labels)
                runner.model.module.roi_head.bbox_head.loss_reid._init_cluster(means.cuda(), stds.cuda())
            else:
                means, stds = get_gaussion_distributation(memory_features[0], memory_labels)
                runner.model.module.roi_head.bbox_head.loss_reid._init_cluster(means.cuda())
        
        runner.model.module.roi_head.bbox_head.loss_reid._update_label(memory_labels)
        
        # if self.part_based_uncertainty:
            # runner.model.module.roi_head.bbox_head.loss_reid._update_blabel(memory_blabels)
            # runner.model.module.roi_head.bbox_head.loss_reid._update_tlabel(memory_tlabels)
            # self.logger.info('bottom pseudo label range: '+ str(memory_blabels.min())+ str(memory_blabels.max()))
            # self.logger.info('top pseudo label range: '+ str(memory_tlabels.min())+ str(memory_tlabels.max()))
        
        self.logger.info('pseudo label range: '+ str(memory_labels.min())+ str(memory_labels.max()))
        self.logger.info("Finished updating pseudo label")
        self.epoch += 1