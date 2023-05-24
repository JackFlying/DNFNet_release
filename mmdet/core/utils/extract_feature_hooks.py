import os.path as osp
import warnings
from xml.etree.ElementTree import TreeBuilder

from mmcv.runner import Hook
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from mmdet.utils import get_dist_info
from mmdet.utils import all_gather_tensor, synchronize
import mmcv
import os
import numpy as np


class ExtractFeatureHook(Hook):
    """Evaluation hook.

    Notes:
        If new arguments are added for EvalHook, tools/test.py may be
    effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, dataloader, cfg, start=None, interval=1, logger=None, pretrained_feature_file=None, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.logger = logger
        self.cfg = cfg
        self.pretrained_feature_file = pretrained_feature_file
        self.alliance_clustering = False
        self.uncertainty_estimation = False
        self.use_part_feats = cfg.USE_PART_FEAT
        self.use_gfn = cfg.USE_GFN
        self.use_feature_std = cfg.USE_STD
        
    
    def before_run(self, runner):
        if not os.path.exists('saved_file'):
            os.mkdir('saved_file')

        self.logger.info('start feature extraction for hybrid memory initialization')
    
        if self.cfg.testing:
            self.dataloader.dataset.id_num = 500
        with torch.no_grad():
            print('feature extract from: ', self.pretrained_feature_file)
            if self.cfg.model.roi_head.bbox_head.type != 'CoLearningHead':
                features, img_ids, person_ids, std_features, features_unnorm = self.extract_features(
                    runner.model, self.dataloader, self.dataloader.dataset, with_path=False, prefix="Extract: ", \
                    pretrained_feature_file=self.pretrained_feature_file)
                # if self.alliance_clustering or self.uncertainty_estimation:
                #     assert features.size(0) == 2 * self.dataloader.dataset.id_num
                #     assert img_ids.size(0) == 2 * self.dataloader.dataset.id_num
                #     assert person_ids.size(0) == 2 * self.dataloader.dataset.id_num
                # else:
                #     assert features.size(0) == self.dataloader.dataset.id_num
                #     assert img_ids.size(0) == self.dataloader.dataset.id_num
                #     assert person_ids.size(0) == self.dataloader.dataset.id_num
                #     assert std_features.size(0) == self.dataloader.dataset.id_num
                if self.cfg.save_features:
                    torch.save(features, os.path.join("saved_file", "features.pth"))
                    torch.save(person_ids, os.path.join("saved_file", "person_ids.pth"))

        if self.use_part_feats:
            bottom_features = torch.load(os.path.join("saved_file","bottom_features.pth"))
            top_features = torch.load(os.path.join("saved_file", "top_features.pth"))
            runner.model.module.roi_head.bbox_head.loss_reid._update_bottom_feature(bottom_features)
            runner.model.module.roi_head.bbox_head.loss_reid._update_top_feature(top_features)
        
        # if self.use_feature_std:
        #     runner.model.module.roi_head.bbox_head.loss_reid._update_top_feature(top_features)
        if std_features is not None:
            runner.model.module.roi_head.bbox_head.loss_reid._update_feature(features, features_unnorm, std_features)
        else:
            runner.model.module.roi_head.bbox_head.loss_reid._update_feature(features)
        runner.model.module.roi_head.bbox_head.loss_reid._init_ids(img_ids)
        torch.save(img_ids, os.path.join("saved_file", "img_ids.pth"))


    @torch.no_grad()
    def extract_features(self,
        model,  # model used for extracting
        data_loader,  # loading data
        dataset,  # dataset with file paths, etc
        cuda=True,  # extract on GPU
        normalize=True,  # normalize feature
        with_path=False,  # return a dict {path:feat} if True, otherwise, return only feat (Tensor)  # noqa
        print_freq=10,  # log print frequence
        save_memory=False,  # gather features from different GPUs all together or in sequence, only for distributed  # noqa
        for_testing=True,
        prefix="Extract: ",
        pretrained_feature_file=None,
    ):

        rank, world_size, is_dist = get_dist_info()
        features = []

        model.eval()
        try:
            if isinstance(model.module.roi_head.bbox_head, nn.ModuleList):
                print('------')
                for i in range(len(model.module.roi_head.bbox_head)):
                    model.module.roi_head.bbox_head[i].proposal_score_max=True
            else:
                print('****')
                model.module.roi_head.bbox_head.proposal_score_max=True
                ori_iou_threshold = model.module.roi_head.test_cfg.nms.iou_threshold
                model.module.roi_head.test_cfg.nms.iou_threshold=2
                ori_max_per_img = model.module.roi_head.test_cfg.max_per_img
                model.module.roi_head.test_cfg.max_per_img=1000
        except:
            assert False, "setting fault"
            pass
        data_iter = iter(data_loader)

        features = None
        if self.alliance_clustering or self.uncertainty_estimation:
            img_ids = torch.zeros(2 * dataset.id_num).long()
            person_ids = torch.zeros(2 * dataset.id_num).long()
        else:
            img_ids = torch.zeros(dataset.id_num).long()
            person_ids = torch.zeros(dataset.id_num).long()

        if pretrained_feature_file is not None:
            pretrain_features = mmcv.load(pretrained_feature_file)

        prog_bar = mmcv.ProgressBar(len(data_loader))
        for i in range(len(data_loader)):
            if self.cfg.testing:
                if i > 60:
                    break
            data = next(data_iter)
            gt_bboxes=data['gt_bboxes'][0]._data[0][0]
            gt_ids = data['gt_labels'][0]._data[0][0][:, 1]
            if self.cfg.data_name == 'PRW':
                gt_img_ids = data['gt_labels'][0]._data[0][0][:, 2]
                gt_person_ids = data['gt_labels'][0]._data[0][0][:, 3]
            elif self.cfg.data_name == 'CUHK_SYSU':
                gt_img_ids = data['gt_labels'][0]._data[0][0][:, 2]
                gt_person_ids = data['gt_labels'][0]._data[0][0][:, 3]

            if pretrained_feature_file is not None:
                if features is None:
                    features = torch.zeros(dataset.id_num, pretrain_features[i][0].shape[1]-5)
                pretrained_gt_bboxes = pretrain_features[i][0][:, :4]
                pretrained_gt_bboxes = torch.from_numpy(pretrained_gt_bboxes)
                scale_factor = data['img_metas'][0].data[0][0]['scale_factor']
                scale_factor = torch.from_numpy(scale_factor).unsqueeze(dim=0)
                pretrained_gt_bboxes = pretrained_gt_bboxes*scale_factor
                diff = (pretrained_gt_bboxes - gt_bboxes).abs().sum()
                if diff>1:
                    print("pretrained boxes don't match")
                    print(diff)
                    exit()
                features[gt_ids] = torch.from_numpy(pretrain_features[i][0][:, 5:])
                img_ids[gt_ids] = gt_img_ids
                person_ids[gt_ids] = gt_person_ids  # For PRW datasets
                prog_bar.update()
                continue
                
            new_data = {'proposals':data['gt_bboxes'], 'img': data['img'], 'img_metas': data['img_metas'], 'use_crop':False}
            result = model(return_loss=False, rescale=False, **new_data)
            reid_features = torch.from_numpy(result[0][0][:, 5:5+256])
            
            if self.use_part_feats:
                bottom_feats = torch.from_numpy(result[0][0][:, 5+256:5+2*256])
                top_feats = torch.from_numpy(result[0][0][:, 5+2*256:5+3*256])
            if self.use_gfn:
                scene_feats = torch.from_numpy(result[0][0][:, 5+3*256:5+3*256+2048])
            if self.use_feature_std:
                std_feats = torch.from_numpy(result[0][0][:, 5+256:5+2*256])
                
            if normalize:
                reid_features_norm = F.normalize(reid_features, p=2, dim=-1)
                if self.use_part_feats:
                    bottom_feats = F.normalize(bottom_feats, p=2, dim=-1)
                    top_feats = F.normalize(top_feats, p=2, dim=-1)
                if self.use_gfn:
                    scene_feats = F.normalize(scene_feats, p=2, dim=-1)

            if features is None:
                if self.alliance_clustering or self.uncertainty_estimation:
                    features = torch.zeros(2 * dataset.id_num, reid_features.shape[1])
                else:
                    features = torch.zeros(dataset.id_num, reid_features.shape[1])
                    features_norm = torch.zeros(dataset.id_num, reid_features_norm.shape[1])
                    if self.use_part_feats:
                        bottom_features = torch.zeros(dataset.id_num, bottom_feats.shape[1])
                        top_features = torch.zeros(dataset.id_num, top_feats.shape[1])
                    if self.use_gfn:
                        scene_features = torch.zeros(dataset.id_num, scene_feats.shape[1])
                    if self.use_feature_std:
                        std_features = torch.zeros(dataset.id_num, std_feats.shape[1])

            #align gt box and predicted box
            result_boxes = torch.from_numpy(result[0][0][:, :4])
            if result_boxes.shape != gt_bboxes.shape:
                # print(model.module.roi_head.test_config.nms.iou_threshold)
                print(result_boxes.shape, gt_bboxes.shape)
                print(result_boxes)
                print(gt_bboxes)
                print(result[0][0][:, :5])
                exit()

            result_boxes = result_boxes.unsqueeze(dim=1)
            gt_bboxes = gt_bboxes.unsqueeze(dim=0)
            diff = (result_boxes - gt_bboxes).abs().sum(dim=-1)
            minis = diff.argmin(dim=-1)
            gt_ids = gt_ids[minis]
            if self.alliance_clustering or self.uncertainty_estimation:
                if self.cfg.testing:
                    dataset.id_num = 500
                gt_ids = torch.cat([gt_ids, gt_ids + dataset.id_num], dim=0)
                gt_img_ids = gt_img_ids.repeat(2)
                gt_person_ids = gt_person_ids.repeat(2)
            
            features[gt_ids] = reid_features
            features_norm[gt_ids] = reid_features_norm
            img_ids[gt_ids] = gt_img_ids
            person_ids[gt_ids] = gt_person_ids
            if self.use_part_feats:
                bottom_features[gt_ids] = bottom_feats
                top_features[gt_ids] = top_feats
            if self.use_gfn:
                scene_features[gt_ids] = scene_feats
            if self.use_feature_std:
                std_features[gt_ids] = std_feats
            
            prog_bar.update()

        #restore model status
        model.train()
        try:
            if isinstance(model.module.roi_head.bbox_head, nn.ModuleList):
                for i in range(len(model.module.roi_head.bbox_head)):
                    model.module.roi_head.bbox_head[i].proposal_score_max=False
            else:
                model.module.roi_head.bbox_head.proposal_score_max=False
                model.module.roi_head.test_cfg.nms.iou_threshold=ori_iou_threshold
                model.module.roi_head.test_cfg.max_per_img=ori_max_per_img
        except:
            assert False, "restore setting fault"
            pass

        synchronize()
        if self.use_part_feats:
            torch.save(bottom_features, os.path.join("saved_file", "bottom_features.pth"))
            torch.save(top_features, os.path.join("saved_file", "top_features.pth"))
            torch.save(top_features, os.path.join("saved_file", "scene_features.pth"))

        if is_dist and cuda:
            # distributed: gather features from all GPUs
            all_features = all_gather_tensor(features.cuda(), save_memory=save_memory)
            all_features = all_features.cpu()[: len(dataset)]
            all_features_norm = all_gather_tensor(features_norm.cuda(), save_memory=save_memory)
            all_features_norm = all_features_norm.cpu()[: len(dataset)]
            all_img_ids = all_gather_tensor(img_ids.cuda(), save_memory=save_memory)
            all_img_ids = all_img_ids.cpu()[: len(dataset)]
            all_person_ids = all_gather_tensor(person_ids.cuda(), save_memory=save_memory)
            all_person_ids = all_person_ids.cpu()[: len(dataset)]
            if self.use_feature_std:
                all_std_features = all_gather_tensor(std_features.cuda(), save_memory=save_memory)
                all_std_features = all_std_features.cpu()[: len(dataset)]
        else:
            all_features = features
            all_img_ids = img_ids
            all_person_ids = person_ids
            all_features_norm = features_norm
            if self.use_feature_std:
                all_std_features = std_features
            else:
                all_std_features = None

        return all_features_norm, img_ids, person_ids, all_std_features, all_features