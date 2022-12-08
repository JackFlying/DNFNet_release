from abc import ABCMeta, abstractmethod

import torch.nn as nn

from ..builder import build_shared_head
import torch
from torch.nn import init

# class NormAwareEmbedding(nn.Module):
#     """
#     Implements the Norm-Aware Embedding proposed in
#     Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
#     """

#     def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=[256, 128]):
#         super(NormAwareEmbedding, self).__init__()
#         self.featmap_names = featmap_names
#         self.in_channels = in_channels
#         self.dim = dim

#         self.projectors = nn.ModuleDict()
#         if len(dim) == 1:
#             indv_dims = self._split_embedding_dim(self.dim[0])
#         else:
#             indv_dims = dim
#         for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
#             proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
#             init.normal_(proj[0].weight, std=0.01)
#             init.normal_(proj[1].weight, std=0.01)
#             init.constant_(proj[0].bias, 0)
#             init.constant_(proj[1].bias, 0)
#             self.projectors[ftname] = proj

#         self.rescaler = []
#         self.bn_feature_seperately = False
#         if self.bn_feature_seperately:
#             for _ in dim:
#                 self.rescaler.append(nn.BatchNorm1d(1, affine=True))
#         else:
#             self.rescaler = nn.BatchNorm1d(1, affine=True)

#     def forward(self, featmaps):
#         """
#         Arguments:
#             featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
#                       featmaps to use
#         Returns:
#             tensor of size (BatchSize, dim), L2 normalized embeddings.
#             tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
#         """
#         assert len(featmaps) == len(self.featmap_names)
#         if len(featmaps) == 1:
#             for k, v in featmaps.items():
#                 pass
#             v = self._flatten_fc_input(v)
#             embeddings = self.projectors[k](v)
#             norms = embeddings.norm(2, 1, keepdim=True)
#             embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
#             norms = self.rescaler(norms).squeeze()
#             return embeddings, norms
#         else:
#             if self.bn_feature_seperately:
#                 embeddings = []
#                 norms = []
#                 for idx_fm, (k, v) in enumerate(featmaps.items()):
#                     v = self._flatten_fc_input(v)
#                     proj_feat = self.projectors[k](v)
#                     norm = proj_feat.norm(2, 1, keepdim=True)
#                     embedding = proj_feat / norm.expand_as(proj_feat).clamp(min=1e-12)
#                     norm = self.rescaler[idx_fm](norm).squeeze()
#                     embeddings.append(embedding)
#                     norms.append(norm)
#                 embeddings = torch.cat(embeddings, dim=1)
#                 norms = torch.cat(norms, dim=1)
#             else:
#                 outputs = []
#                 for k, v in featmaps.items():
#                     v = self._flatten_fc_input(v)
#                     proj_feat = self.projectors[k](v)
#                     outputs.append(proj_feat)
#                 embeddings = torch.cat(outputs, dim=1)
#                 norms = embeddings.norm(2, 1, keepdim=True)
#                 embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
#                 norms = self.rescaler(norms).squeeze()
#             return embeddings, norms

#     def _flatten_fc_input(self, x):
#         if x.ndimension() == 4:
#             assert list(x.shape[2:]) == [1, 1]
#             return x.flatten(start_dim=1)
#         return x

#     def _split_embedding_dim(self, dim):
#         parts = len(self.in_channels)
#         tmp = [dim // parts] * parts
#         if sum(tmp) == dim:
#             return tmp
#         else:
#             res = dim % parts
#             for i in range(1, res + 1):
#                 tmp[-i] += 1
#             assert sum(tmp) == dim
#             return tmp

# class SEAttention(nn.Module):
#     def __init__(self, channel=512, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


class SafeBatchNorm1d(torch.nn.BatchNorm1d):
    """
    Handles case where batch size is 1.
    """
    def forward(self, x):
        # If batch size is 1, use running batch statistics
        if (x.size(0) == 1) and self.training:
            self.eval()
            y = super().forward(x)
            self.train()
        # Otherwise compute statistics for this batch as normal
        else:
            y = super().forward(x)
        return y

class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.

    For SeqNeXt, we do not use embedding norms as scores by default, but retain the option
    mainly for comparitive purposes.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256, norm_type='batchnorm'):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        if norm_type == 'layernorm':
            norm_layer = nn.LayerNorm
        elif norm_type == 'batchnorm':
            norm_layer = SafeBatchNorm1d

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), norm_layer(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        # Affine True by default for both BatchNorm1d, LayerNorm
        self.rescaler = norm_layer(1)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = list(featmaps.items())[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            if self.rescaler is None:
                norms = norms.squeeze()
            else:
                norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            if self.rescaler is None:
                norms = norms.squeeze()
            else:
                norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp

class BaseRoIHead(nn.Module, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_gfn=False,
                 use_RoI_Align_feat=False,
                 use_part_feat=True,
                 scene_emb_size=56):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)
        self.bbox_head_cfg = bbox_head
        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()

        self.use_gfn = use_gfn
        self.use_RoI_Align_feat = use_RoI_Align_feat
        self.use_part_feat = use_part_feat
        self.scene_emb_size = scene_emb_size
        
        self.embedding_head = NormAwareEmbedding(
            featmap_names=["feat_res4", "feat_res5"],
            in_channels=[1024, 2048],
            dim=2048,
            norm_type='batchnorm'
        )
        # self.use_global_Local_context = False
        # self.sampler_num = None
        # self.cxt_feat_len = 1024
        # self.psn_feat_len = 2048
        # self.feat_res4_len = 1024
        # self.feat_res5_len = 2048
        # # self.reid_len = 2048
        # # self.cat_reid_len = self.reid_len + self.cxt_feat_len + self.reid_len
        # self.out_dim_reg = 4
        # self.cxt_feat_extractor_scenario = nn.Sequential(
        #     nn.Conv2d(1024, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, self.cxt_feat_len, 3, 1, 1),
        #     nn.BatchNorm2d(self.cxt_feat_len),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveMaxPool2d(1)
        # )
        # self.cxt_feat_extractor_psn = nn.Sequential(
        #     nn.Conv2d(2048, 256, 1, 1, 0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 1, 1, 0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, self.psn_feat_len, 3, 1, 1),
        #     nn.BatchNorm2d(self.psn_feat_len),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveMaxPool2d(1)
        # )
        # self.embedding_head = NormAwareEmbedding(
        #     featmap_names=["feat_res4", "feat_res5"],
        #     in_channels=[self.feat_res4_len, self.feat_res5_len],
        #     dim=[256],
        # )
        # self.fc_reg = nn.Sequential(nn.Linear(self.feat_res5_len, self.out_dim_reg),
        #                 nn.BatchNorm1d(self.out_dim_reg)
        # )

    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""
        pass

    async def async_simple_test(self, x, img_meta, **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        pass

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pass
