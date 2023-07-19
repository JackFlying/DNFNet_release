from re import T
from tkinter import N
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import torch.nn.functional as F
import torch.nn as nn

@HEADS.register_module()
class ReidRoIHeadDNFNet(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        use_crop = kwargs['use_crop']
        crop_feats_list = self._crop_forward(kwargs, gt_bboxes)

        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                if use_crop:
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x],
                        crop_feats=crop_feats_list[i])
                else:
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x],
                        crop_feats=None)
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, **kwargs)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        return losses

    def _crop_forward(self, kwargs, gt_bboxes):
        use_crop = kwargs['use_crop']
        if use_crop:
            crop_feats = kwargs['crop_feats']
            crop_feats1 = []
            crop_feats2 = []
            for i in range(len(crop_feats)):
                crop_feat1 = F.adaptive_max_pool2d(crop_feats[i], 1)
                crop_feats1.append(crop_feat1)
                if self.with_shared_head:
                    crop_feat2 = self.shared_head(crop_feats[i])   # [N, 2048, 7, 7]
                    crop_feat2 = F.adaptive_max_pool2d(crop_feat2, 1)   # [N, 2048, 1, 1]
                    crop_feats2.append(crop_feat2)
            crop_feats1 = torch.cat(crop_feats1, dim=0)
            crop_feats2 = torch.cat(crop_feats2, dim=0)
            crop_feats3 = self.bbox_head.crop_forward(crop_feats1, crop_feats2)
            splits = [gt_bboxes[i].shape[0] for i in range(len(gt_bboxes))]
            crop_feats_list = crop_feats3.split(splits)
            return crop_feats_list
        else:
            return None

    def _bbox_forward(self, x, rois, labels=None, bbox_targets=None, test=False):
        """Box head forward function used in both training and testing."""
        # part_feats, part_feats1, RoI_Align_feat = None, None, None
        # bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)   # [N, 1024, 14, 14], [14, 14]表示[height, width]
        # bbox_feats1 = F.adaptive_max_pool2d(bbox_feats, 1).squeeze(-1).squeeze(-1)  # [N, 1024, 1, 1]

        # if self.use_part_feat:
        #     part_feats = torch.nn.AdaptiveAvgPool2d((4, 1))(bbox_feats)   # [N, 1024, 2, 1]
        #     part_feats1 = [part_feats[:, :, i:i+1] for i in range(part_feats.shape[2])] # 2 * [N, 1024, 1, 1]

        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)   # [N, 2048, 7, 7]
        #     if self.use_part_feat:
        #         part_feats = [self.shared_head(part_feats1[i]) for i in range(len(part_feats1))]
        #     if self.use_RoI_Align_feat:
        #         RoI_Align_feat = bbox_feats.detach()    # visualize heat map
        #     bbox_feats = F.adaptive_max_pool2d(bbox_feats, 1)   # [N, 2048, 1, 1]

        RoI_Align_feat = None
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)   # [N, 1024, 14, 14], [14, 14]表示[height, width]
        bbox_feats1 = F.adaptive_max_pool2d(bbox_feats, 1).squeeze(-1).squeeze(-1)  # [N, 1024, 1, 1]
        if self.use_part_feat:
            part_num = 2
            # part_height = bbox_feats.shape[2] // part_num
            part_height = bbox_feats.shape[2] - 1
            part_feats_list = [bbox_feats[:, :, i:i+part_height, :] for i in range(part_num)]   # N, 1024, 7, 14]
            part_feats1_list = [F.adaptive_max_pool2d(x, 1) for x in part_feats_list]  # 2 * [N, 1024, 1, 1]
        
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)   # [N, 2048, 7, 7]
            bbox_feats = F.adaptive_max_pool2d(bbox_feats, 1)   # [N, 2048, 1, 1]
            if self.use_part_feat:
                part_feats_list = [self.shared_head(x) for x in part_feats_list]
                part_feats_list = [F.adaptive_max_pool2d(x, 1) for x in part_feats_list]
            if self.use_RoI_Align_feat:
                RoI_Align_feat = bbox_feats.detach()    # visualize heat map

        scene_emb, scene_emb1, scene_emb2, gfn_losses = None, None, None, torch.tensor(0.)
        # if self.use_gfn:
        #     scene_emb1 = F.adaptive_max_pool2d(x[0], 1).squeeze(-1).squeeze(-1) # x[0]=[N, 1024, H, W] => [N, 1024, 1, 1]
        #     if self.with_shared_head:
        #         scene_emb2 = F.adaptive_max_pool2d(x[0], self.scene_emb_size) # [N, 1024, scene_emb_size, scene_emb_size]
        #         scene_emb2 = self.shared_head(x[0])
        #         scene_emb2 = F.adaptive_max_pool2d(scene_emb2, 1).squeeze(-1).squeeze(-1) # [N, 2048, 1, 1]
        #         scene_emb = self.embedding_head({'feat_res4':scene_emb1, 'feat_res5':scene_emb2})[0]
    
        # cls_score, bbox_pred, id_pred, part_id_pred, _, query_embed = self.bbox_head(bbox_feats1, bbox_feats, part_feats1, part_feats, scene_emb1, scene_emb2, labels, rois, bbox_targets) # [N, 256]
        cls_score, bbox_pred, id_pred, part_id_pred, _, query_embed = self.bbox_head(bbox_feats1, bbox_feats, part_feats1_list, part_feats_list, scene_emb1, scene_emb2, labels, rois, bbox_targets) # [N, 256]
        # if self.use_gfn and self.training:
        #     gfn_losses = self.bbox_head.gfn_forward(scene_emb, query_embed, labels)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, id_pred=id_pred, \
                            RoI_Align_feat=RoI_Align_feat, part_id_pred=part_id_pred, scene_emb=scene_emb, gfn_losses=gfn_losses)
        
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, **kwargs):
        """Run forward function and calculate loss for box head in training."""
        self.sampler_num = [res.pos_bboxes.shape[0] + res.neg_bboxes.shape[0] for res in sampling_results]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, **kwargs)
        labels = bbox_targets[0]
        # print("labels", labels.shape)
        bbox_results = self._bbox_forward(x, rois, labels, bbox_targets)
        # if self.use_global_Local_context:
        #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score_logit'],
        #                                     bbox_results['bbox_pred'],
        #                                     bbox_results['id_pred'], 
        #                                     rois,
        #                                     *bbox_targets,
        #                                     **kwargs)
        # else:
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        bbox_results['id_pred'], 
                                        bbox_results['part_id_pred'],
                                        rois,
                                        # crop_targets=None,
                                        *bbox_targets,
                                        **kwargs)
        loss_bbox.update(gfn_losses=bbox_results['gfn_losses'])
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results
    
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           **kwargs):
        """Test only det bboxes without augmentation."""
        if kwargs['use_crop']:
            crop_feats_list = self._crop_forward(kwargs, proposals)
            crop_feats = crop_feats_list[0]
        else:
            crop_feats = None

        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois, test=True)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        id_pred = bbox_results['id_pred']
        RoI_Align_feat = bbox_results['RoI_Align_feat']
        part_id_pred = bbox_results['part_id_pred']
        scene_emb = bbox_results['scene_emb']

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        bbox_pred = bbox_pred.split(num_proposals_per_img,0) if bbox_pred is not None else [None, None]
        id_pred = id_pred.split(num_proposals_per_img, 0)
        
        if scene_emb is not None:
            scene_emb = scene_emb.repeat(num_proposals_per_img[0], 1)
        
        if RoI_Align_feat is not None:
            RoI_Align_feat = RoI_Align_feat.split(num_proposals_per_img, 0)

        if part_id_pred is not None:
            part_id_pred = part_id_pred.split(num_proposals_per_img, 0)

        if crop_feats is not None:
            crop_feats = crop_feats.split(num_proposals_per_img, 0)

        if self.bbox_head_cfg.type == 'CoLearningHead':
            id_pred2 = bbox_results['id_pred2']
            id_pred2 = id_pred2.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        if self.bbox_head_cfg.type == 'CoLearningHead':
            for i in range(len(proposals)):
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    id_pred[i],
                    id_pred2[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
        else:
            for i in range(len(proposals)):
                if scene_emb is not None:
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        id_pred[i],
                        # RoI_Align_feat[i].view(RoI_Align_feat[i].shape[0], -1),  # [2048, 7, 7]
                        torch.cat([part_id_pred[i], scene_emb], dim=1),
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg)
                else:
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        id_pred[i],
                        # RoI_Align_feat[i].view(RoI_Align_feat[i].shape[0], -1),  # [2048, 7, 7]
                        part_id_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg)
                
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
        return det_bboxes, det_labels

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False, 
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, **kwargs)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
