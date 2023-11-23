import os
from scipy.io import loadmat
import os.path as osp
import numpy as np
import pickle
import re
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

import os
import mmcv
import torch
from mmcv import Config
from pycocotools.coco import COCO
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import Compose
from ps_model import load_model
from vis_search import vis_search_result
from __init__ import *


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def set_box_pid(boxes, box, pids, pid):
    for i in range(boxes.shape[0]):
        if np.all(boxes[i] == box):
            pids[i] = pid
            return
    print("Person: %s, box: %s cannot find in images." % (pid, box))

def image_path_at(data_path, image_index, i):
    image_path = osp.join(data_path, image_index[i])
    assert osp.isfile(image_path), "Path does not exist: %s" % image_path
    return image_path

def load_image_index(root_dir, db_name):
    """Load the image indexes for training / testing."""
    # Test images
    test = loadmat(osp.join(root_dir, "annotation", "pool.mat"))
    test = test["pool"].squeeze()
    test = [str(a[0]) for a in test]
    if db_name == "psdb_test":
        return test

    # All images
    all_imgs = loadmat(osp.join(root_dir, "annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    all_imgs = [str(a[0][0]) for a in all_imgs]

    # Training images = all images - test images
    train = list(set(all_imgs) - set(test))
    train.sort()
    return train

def _get_cam_id(im_name):
        match = re.search('c\d', im_name).group().replace('c', '')
        return int(match)

def load_probes(root):
    query_info = osp.join(root, 'query_info.txt')
    with open(query_info, 'r') as f:
        raw = f.readlines()

    probes = []
    pname_to_attribute = {}
    probes_name_list = []
    for line in raw:
        linelist = line.split(' ')
        pid = int(linelist[0])
        x, y, w, h = float(linelist[1]), float(linelist[2]), float(linelist[3]), float(linelist[4])
        roi = np.array([x, y, x + w, y + h]).astype(np.int32)
        roi = np.clip(roi, 0, None)  # several coordinates are negative
        im_name = linelist[5][:-1] + '.jpg'
        # probes.append({'im_name': im_name,
        #                 'boxes': roi[np.newaxis, :],
        #                 # Useless. Can be set to any value.
        #                 'gt_pids': np.array([pid]),
        #                 'flipped': False,
        #                 'cam_id': _get_cam_id(im_name)
        #                 })
        pname_to_attribute[im_name] = {
                        'boxes': roi[np.newaxis, :],
                        'gt_pids': np.array([pid]),
                        'flipped': False,
                        'cam_id': _get_cam_id(im_name)
                        }
        probes_name_list.append(im_name)
    return pname_to_attribute, probes_name_list

def gt_roidbs(root):
    imgs = loadmat(osp.join(root, 'frame_test.mat'))['img_index_test']
    imgs = [img[0][0] + '.jpg' for img in imgs]

    gt_roidb = []
    for im_name in imgs:
        anno_path = osp.join(root, 'annotations', im_name)
        anno = loadmat(anno_path)
        box_key = 'box_new'
        if box_key not in anno.keys():
            box_key = 'anno_file'
        if box_key not in anno.keys():
            box_key = 'anno_previous'

        rois = anno[box_key][:, 1:]
        ids = anno[box_key][:, 0]
        rois = np.clip(rois, 0, None)  # several coordinates are negative

        assert len(rois) == len(ids)

        rois[:, 2:] += rois[:, :2]
        # num_objs = len(rois)
        # overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # overlaps[:, 1] = 1.0
        # overlaps = csr_matrix(overlaps)
        gt_roidb.append({
            'im_name': im_name,
            'boxes': rois.astype(np.int32),
            'gt_pids': ids.astype(np.int32),
            'flipped': False,
            'cam_id': _get_cam_id(im_name)
            # 'gt_overlaps': overlaps
        })
    return gt_roidb

class PRW_UNSUPDataset():
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    def __init__(self,
                 ann_file,
                 pipeline,
                 query_test_pipeline=None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 probes_name_list=None):
        
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.probes_name_list = probes_name_list

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        self.data_infos = self.load_annotations(self.ann_file)
        # filter data infos if classes are customized
        if self.custom_classes:
            self.data_infos = self.get_subset_by_classes()

        self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)
        if query_test_pipeline is None:
            self.query_test_pipeline = None
        else:
            self.query_test_pipeline = Compose(query_test_pipeline)

        # query mode
        self.query_mode=False

        #unsupervised: unique labels
        self.generate_unique_ids()

    def generate_unique_ids(self):
        self.id_num = len(self.coco.anns)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        """
        if classes is None:
            cls.custom_classes = False
            return cls.CLASSES

        cls.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            if info['file_name'] in self.probes_name_list:
                info['filename'] = info['file_name']
                data_infos.append(info)
        print("data_infos", len(data_infos))
        print("probes_name_list", len(self.probes_name_list))
        return data_infos

    def pre_pipeline(cfg, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = cfg.img_prefix
        results['seg_prefix'] = cfg.seg_prefix
        results['proposal_file'] = cfg.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        
    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            # print("ann", ann)
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            #person_id = ann['person_id']
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # gt_labels.append([self.cat2label[ann['category_id']], person_id])
                gt_labels.append([self.cat2label[ann['category_id']], ann['id'], ann['image_id'], ann['person_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0, 2), dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann    
        
    def __call__(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

def get_prw_dataset_info(info, det_thresh=0.5, data_root = '/home/linhuadong/dataset/PRW'):
    pname_to_attribute, probes_name_list = load_probes(data_root)  # 2057
    gallery_set = gt_roidbs(data_root)  # 6112, probe在gallery当中

    with open(os.path.join(info["root_dir"], 'results_1000.pkl'), 'rb') as fid:
        all_dets = pickle.load(fid) # gallery每张图片的检测和重识别结果

    gallery_det, gallery_feat = [], []
    for det in all_dets:    # 6112
        if det[0].shape[0] > 0:
            det_ = det[0][:, :5]
            feat = normalize(det[0][:, 5:5+256], axis=1)
        else:
            det_ = np.zeros((0, 5), dtype=np.float32)
            feat = det[0][:, 5:5+256]
        gallery_det.append(det_)
        gallery_feat.append(feat)

    gt_roidb = gallery_set
    name_to_det_feat = {}
    for gt, det, feat in zip(gt_roidb, gallery_det, gallery_feat):
        name = gt['im_name']
        pids = gt['gt_pids']
        cam_id = gt['cam_id']
        scores = det[:, 4].ravel()
        inds = np.where(scores >= det_thresh)[0]
        if len(inds) > 0:
            name_to_det_feat[name] = (det[inds], feat[inds], pids, cam_id)

    cfg = Config.fromfile(info["config"])
    PRW_Dataset = PRW_UNSUPDataset(
                            cfg.data.test.ann_file,
                            cfg.data.test.pipeline,
                            cfg.data.test.query_test_pipeline,
                            classes=None,
                            data_root=cfg.data_root,
                            img_prefix=cfg.data.test.img_prefix,
                            seg_prefix=None,
                            proposal_file=cfg.data.test.proposal_file,
                            test_mode=True,
                            filter_empty_gt=True,
                            probes_name_list=probes_name_list
                            )
    return PRW_Dataset, pname_to_attribute, gt_roidb, name_to_det_feat

def get_prw_data(PRW_Dataset, pname_to_attribute, idx_):
    data = PRW_Dataset(idx=idx_)
    data['img'][0] = data['img'][0].unsqueeze(0).contiguous()
    data['img_metas'][0]._data = [[data['img_metas'][0]._data]]
    
    probe_imname = data['img_metas'][0]._data[0][0]['ori_filename']
    attribute = pname_to_attribute[probe_imname]
    probe_roi = attribute['boxes']
    scale_factor = data['img_metas'][0]._data[0][0]['scale_factor']
    rescale_gt_bboxes = probe_roi * scale_factor
    data['proposals'] = [DC([[torch.tensor(rescale_gt_bboxes).float()]])]
    return data

def search_performance_prw(result, data, pname_to_attribute, name_to_det_feat, gt_roidb, ignore_cam_id=True):

    probe_imname = data['img_metas'][0]._data[0][0]['ori_filename']
    attribute = pname_to_attribute[probe_imname]
    probe_roi = attribute['boxes']
    probe_pid = attribute['gt_pids']
    probe_cam = attribute['cam_id']

    det = result[0][0][:, :5]
    feat_p = normalize(result[0][0][:, 5:5+256], axis=1).ravel()

    topk = [1, 5, 10]
    image_root = '/home/linhuadong/dataset/PRW/frames'
    # ret = {'image_root': image_root, 'results': []}

    y_true, y_score = [], []
    imgs, rois = [], []
    count_gt, count_tp = 0, 0

    gallery_imgs = []
    for x in gt_roidb:
        if probe_pid in x['gt_pids'] and x['im_name'] != probe_imname:
            gallery_imgs.append(x)
    # find image name and corresponding instance box with the same identity

    probe_gts = {}
    for item in gallery_imgs:
        probe_gts[item['im_name']] = item['boxes'][item['gt_pids'] == probe_pid]

    # Construct gallery set for this probe
    if ignore_cam_id:
        gallery_imgs = []
        for x in gt_roidb:
            if x['im_name'] != probe_imname:
                gallery_imgs.append(x)
    else:
        gallery_imgs = []
        for x in gt_roidb:
            if x['im_name'] != probe_imname and x['cam_id'] != probe_cam:
                gallery_imgs.append(x)

    # 1. Go through all gallery samples. for item in testset.targets_db: Gothrough the selected gallery
    for item in gallery_imgs:
        gallery_imname = item['im_name']
        # some contain the probe (gt not empty), some not
        count_gt += (gallery_imname in probe_gts)
        # compute distance between probe and gallery dets
        if gallery_imname not in name_to_det_feat:
            continue
        det, feat_g, _, _ = name_to_det_feat[gallery_imname]
        # get L2-normalized feature matrix NxD
        assert feat_g.size == np.prod(feat_g.shape[:2])
        feat_g = feat_g.reshape(feat_g.shape[:2])

        scores = det[:, 4]
        feat_g = scores[:, np.newaxis] * feat_g
        # compute cosine similarities
        sim = feat_g.dot(feat_p).ravel()
        # assign label for each det
        label = np.zeros(len(sim), dtype=np.int32)
        if gallery_imname in probe_gts:
            gt = probe_gts[gallery_imname].ravel()
            w, h = gt[2] - gt[0], gt[3] - gt[1]
            iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
            inds = np.argsort(sim)[::-1]
            sim = sim[inds]
            det = det[inds]
            # only set the first matched det as true positive
            for j, roi in enumerate(det[:, :4]):
                if compute_iou(roi, gt) >= iou_thresh:
                    label[j] = 1
                    count_tp += 1
                    break

        y_true.extend(list(label))
        y_score.extend(list(sim))
        imgs.extend([gallery_imname] * len(sim))
        rois.extend(list(det))

    # 2. Compute AP for this probe (need to scale by recall rate)
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    assert count_tp <= count_gt
    recall_rate = count_tp * 1.0 / count_gt
    ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
    inds = np.argsort(y_score)[::-1]
    y_score = y_score[inds]
    y_true = y_true[inds]
    acc = [min(1, sum(y_true[:k])) for k in topk]
    # 4. Save result for JSON dump
    new_entry = {
        'probe_img': str(probe_imname),
        'probe_roi': list(probe_roi.squeeze()),
        'probe_gt': probe_gts,
        'ap':ap, 
        'acc':acc,
        'pid':probe_pid,
        'gallery': [],
        'image_root': image_root
    }
    # only save top-10 predictions
    for k in range(10):
        new_entry['gallery'].append({
                'img': str(imgs[inds[k]]),
                'roi': list(rois[inds[k]]),
                'score': float(y_score[k]),
                'correct': int(y_true[k]),
            }
        )
    # ret['results'].append(new_entry)
    print(y_true[:5])
    return new_entry

def main():
    info_sota = get_info_sota()
    model_prw = load_model(info_sota['PRW'])
    PRW_Dataset, pname_to_attribute, gt_roidb, name_to_det_feat_prw = get_prw_dataset_info(info_sota['PRW'])
    idx = 2
    data = get_prw_data(PRW_Dataset, pname_to_attribute, idx)
    import ipdb;    ipdb.set_trace()
    with torch.no_grad():
        result = model_prw(return_loss=False, rescale=True, **data)
    entry = search_performance_prw(result, data, pname_to_attribute, name_to_det_feat_prw, gt_roidb)
    vis_search_result(entry)
    

if __name__ == "__main__":
    idx_ = 2    # idx \in [0, PRW_Dataset.data_infos)
    main()