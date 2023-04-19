# Written by Yixiao Ge

import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.dist_utils import (
    broadcast_tensor,
    broadcast_value,
    get_dist_info,
    synchronize,
)
from .dbscan_context import label_generator_dbscan_context, label_generator_dbscan_context_single 
from .hdbscan_context import label_generator_hdbscan_context, label_generator_hdbscan_context_single 
from .dbscan_context_eps import label_generator_dbscan_context_eps 
from .dbscan_context_eps_all import label_generator_dbscan_context_eps_all
from .dbscan_context_eps_all_weight import label_generator_dbscan_context_eps_all_weight
from .dbscan_context_kmeans import label_generator_dbscan_context_kmeans
from .infomap_context import label_generator_infomap_context, label_generator_infomap_context_single 
from .dbscan import label_generator_dbscan, label_generator_dbscan_single  # noqa
from .kmeans import label_generator_kmeans
# from .FINCH_context import label_generator_FINCH_context
from .FINCH_context_SpCL import label_generator_FINCH_context_SpCL, label_generator_FINCH_context
from .FINCH_context_SpCL_plus import label_generator_FINCH_context_SpCL_Plus

class LabelGenerator(object):
    """Pseudo Label Generator."""

    __factory = {
        "dbscan_context_kmeans": label_generator_dbscan_context_kmeans,
        "dbscan_context_eps": label_generator_dbscan_context_eps,
        "dbscan_context_eps_all": label_generator_dbscan_context_eps_all,
        "dbscan_context_eps_all_weight": label_generator_dbscan_context_eps_all_weight,
        "dbscan_context": label_generator_dbscan_context,
        "hdbscan_context": label_generator_hdbscan_context,
        "infomap_context": label_generator_infomap_context,
        "FINCH_context":label_generator_FINCH_context,
        "FINCH_context_SpCL":label_generator_FINCH_context_SpCL,
        "FINCH_context_SpCL_Plus":label_generator_FINCH_context_SpCL_Plus,
        "FINCH_context_SpCL_Hierarchical":label_generator_FINCH_context_SpCL_Plus,
        "dbscan": label_generator_dbscan,
        "kmeans": label_generator_kmeans,
    }

    def __init__(
        self, cfg, data_loaders, verbose=True  # list of models, e.g. MMT has two models
    ):
        super(LabelGenerator, self).__init__()


        self.cfg = cfg
        self.verbose = verbose

        self.cluster_type = self.cfg.PSEUDO_LABELS.cluster

        self.num_classes = []
        self.indep_thres = []

        if self.cfg.PSEUDO_LABELS.cluster_num is not None:
            # for kmeans
            self.num_classes = self.cfg.PSEUDO_LABELS.cluster_num
        
        self.data_loaders = data_loaders
        self.datasets = [i.dataset for i in data_loaders]

        self.rank, self.world_size, _ = get_dist_info()
        self.eps = None

    @torch.no_grad()
    def __call__(self, cuda=True, memory_features=None, image_inds=None, epoch=0, eps=None, **kwargs):

        all_labels, all_blabels, all_tlabels = [], [], []
        all_centers = []
        

        for idx, (data_loader, dataset) in enumerate(
            zip(self.data_loaders, self.datasets)
        ):

            # clustering
            try:
                indep_thres = self.indep_thres[idx]
            except Exception:
                indep_thres = None
            try:
                num_classes = self.num_classes[idx]
            except Exception:
                num_classes = None

            assert isinstance(memory_features, list)
            all_features = memory_features[idx]
            if image_inds is not None:
                all_inds = image_inds
            else:
                all_inds = None

            if self.cfg.PSEUDO_LABELS.norm_feat:
                if isinstance(all_features, list):
                    all_features = [F.normalize(f, p=2, dim=1) for f in all_features]
                else:
                    all_features = F.normalize(all_features, p=2, dim=1)
            if self.rank == 0:
                # clustering only on GPU:0
                if self.cluster_type in ['dbscan_context_eps', 'dbscan_context_eps_all', 'dbscan_context_eps_all_weight']:
                    labels, centers, num_classes, indep_thres, tmp_eps = self.__factory[
                        self.cluster_type
                    ](
                        self.cfg,
                        all_features,
                        num_classes=num_classes,
                        cuda=cuda,
                        indep_thres=indep_thres,
                        all_inds=all_inds,
                        epoch=epoch,
                        eps=self.eps
                    )
                    self.eps = tmp_eps
                else:
                    labels, centers, num_classes, indep_thres, blabels, tlabels = self.__factory[
                        self.cluster_type
                    ](
                        kwargs['cfg'],
                        all_features,
                        num_classes=num_classes,
                        cuda=cuda,
                        indep_thres=indep_thres,
                        all_inds=all_inds,
                        epoch=epoch,
                        iters=kwargs.get('iters', [0, 1, 2])
                    )

                if self.cfg.PSEUDO_LABELS.norm_center:
                    centers = F.normalize(centers, p=2, dim=1)

            synchronize()

            # broadcast to other GPUs
            if self.world_size > 1:
                num_classes = int(broadcast_value(num_classes, 0))
                if (
                    self.cfg.PSEUDO_LABELS == "dbscan"
                    and len(self.cfg.PSEUDO_LABELS.eps) > 1
                ):
                    # use clustering reliability criterion
                    indep_thres = broadcast_value(indep_thres, 0)
                if self.rank > 0:
                    labels = torch.arange(len(dataset)).long()
                    centers = torch.zeros((num_classes, all_features.size(-1))).float()
                labels = broadcast_tensor(labels, 0)
                centers = broadcast_tensor(centers, 0)

            try:
                self.indep_thres[idx] = indep_thres
            except Exception:
                self.indep_thres.append(indep_thres)
            try:
                self.num_classes[idx] = num_classes
            except Exception:
                self.num_classes.append(num_classes)

            all_labels.append(labels.tolist())
            all_centers.append(centers)
            if blabels is not None:
                all_tlabels.append(tlabels.tolist())
                all_blabels.append(blabels.tolist())

        self.cfg.PSEUDO_LABELS.cluster_num = self.num_classes

        for label in all_labels:
                self.print_label_summary(label)


        return all_labels, all_centers, all_blabels, all_tlabels

    def print_label_summary(self, pseudo_labels):
        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label] += 1
        if -1 in index2label.keys():
            unused_ins_num = index2label.pop(-1)
        else:
            unused_ins_num = 0
        index2label = np.array(list(index2label.values()))
        clu_num = (index2label > 1).sum()
        unclu_ins_num = (index2label == 1).sum()
        print(
            f"{clu_num} clusters, "
            f"{unclu_ins_num} un-clustered instances, "
            f"{unused_ins_num} unused instances\n"
        )
