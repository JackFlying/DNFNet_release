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

    def __init__(self):
        self.upload_root = "/home/linhuadong/DNFNet/tools/system/upload"
        
        img_norm_cfg = dict(
            mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
        
        self.pipeline = Compose([
                # dict(type='LoadProposals', num_max_proposals=None),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(1500, 900),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='ToTensor', keys=['proposals']),
                        dict(type='Collect', keys=['img', 'proposals']),
                    ]
                )
            ]
        )

    def __call__(self, file):
        results = {}
        results['img_info'] = {'file_name': file, 
                                'id': -1, 
                                'width': -1, 
                                'height': -1, 
                                'filename': osp.join(self.upload_root, file)
                                }

        self.file_client = mmcv.FileClient(**dict(backend='disk'))
        filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag='color')

        h, w, c = img.shape
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_info']['width'] = w
        results['img_info']['height'] = h
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        results['proposals'] = np.array([[0, 0, w, h]], dtype=np.float32)
        results['bbox_fields'] = ['proposals']

        return self.pipeline(results)

def get_my_prw_dataset_info(info, det_thresh=0.5, data_root = '/home/linhuadong/dataset/PRW'):
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

    PRW_Dataset = PRW_UNSUPDataset()
    return PRW_Dataset, gt_roidb, name_to_det_feat

def get_input_prw_data(PRW_Dataset, file_name):
    data = PRW_Dataset(file_name)
    data['img'][0] = data['img'][0].unsqueeze(0).contiguous()
    data['img_metas'][0]._data = [[data['img_metas'][0]._data]]
    data['proposals'] = [DC([data['proposals']])]
    return data

def search_performance_input_prw(result, data, name_to_det_feat, gt_roidb):
    # import ipdb;    ipdb.set_trace()
    probe_imname = data['img_metas'][0]._data[0][0]['ori_filename'].split('/')[-1]
    probe_roi = data['proposals'][0]._data[0][0]
    scale_factor = data['img_metas'][0]._data[0][0]['scale_factor']
    probe_roi = probe_roi / scale_factor

    det = result[0][0][:, :5]
    feat_p = normalize(result[0][0][:, 5:5+256], axis=1).ravel()

    # topk = [1, 5, 10]
    image_root = '/home/linhuadong/dataset/PRW/frames'
    image_root2 = '/home/linhuadong/DNFNet/tools/system/upload'

    y_score = []
    imgs, rois = [], []
    # count_gt, count_tp = 0, 0 

    gallery_imgs = []
    for x in gt_roidb:
        gallery_imgs.append(x)
    # 1. Go through all gallery samples. for item in testset.targets_db: Gothrough the selected gallery
    for item in gallery_imgs:
        gallery_imname = item['im_name']
        # some contain the probe (gt not empty), some not
        # count_gt += (gallery_imname in probe_gts)
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

        y_score.extend(list(sim))
        imgs.extend([gallery_imname] * len(sim))
        rois.extend(list(det))

    inds = np.argsort(y_score)[::-1]

    # 4. Save result for JSON dump
    new_entry = {
        'probe_img': str(probe_imname),
        'probe_roi': list(probe_roi.squeeze()),
        'probe_gt': None,
        'ap':None, 
        'acc':None,
        'pid':-1,
        'gallery': [],
        'image_root': image_root,
        'image_root2':image_root2,
    }
    # only save top-10 predictions
    for k in range(10):
        new_entry['gallery'].append({
                'img': str(imgs[inds[k]]),
                'roi': list(rois[inds[k]]),
                'score': float(y_score[k]),
                'correct': int(2),
            }
        )
    return new_entry

def main():
    info_sota = get_info_sota()
    model_prw = load_model(info_sota['PRW'])
    PRW_Dataset, gt_roidb, name_to_det_feat_prw = get_my_prw_dataset_info(info_sota['PRW'])
    file_name = "0479_c7_s3_016471.jpg"
    data = get_input_prw_data(PRW_Dataset, file_name)
    with torch.no_grad():
        result = model_prw(return_loss=False, rescale=True, **data)
    entry = search_performance_input_prw(result, data, name_to_det_feat_prw, gt_roidb)
    vis_search_result(entry)

if __name__ == "__main__":
    main()