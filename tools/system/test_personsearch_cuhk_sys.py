import os.path as osp
import mmcv
import torch
import numpy as np
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score
from mmdet.datasets import (build_dataloader, build_dataset)
from scipy.io import loadmat
from tqdm import tqdm
import sys
sys.path.append('/home/linhuadong/DNFNet/tools')
from person_search.psdb import PSDB
from ps_model import load_model
from __init__ import *


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def get_cuhk_dataset_info(info):
    cfg = Config.fromfile(info["config"])
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    query_dataset = build_dataset(cfg.data.test)
    query_dataset.load_query()
    query_data_loader = build_dataloader(
        query_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    gallery_det = mmcv.load(osp.join(info["root_dir"], "gallery_detections.pkl"))
    gallery_feat = mmcv.load(osp.join(info["root_dir"], "gallery_features.pkl"))
    
    threshold = 0.5
    psdb_dataset = PSDB("psdb_test", cfg.data_root)
    name_to_det_feat = {}
    for name, det, feat in zip(psdb_dataset.image_index, gallery_det, gallery_feat):
        scores = det[:, 4].ravel()
        inds = np.where(scores >= threshold)[0]
        if len(inds) > 0:
            name_to_det_feat[name] = (det[inds], feat[inds])
    
    return query_data_loader, psdb_dataset, name_to_det_feat

def get_cuhk_data(query_data_loader, idx):
    data = query_data_loader.dataset[idx]
    data['img'][0] = data['img'][0].unsqueeze(0).contiguous()
    data['img_metas'][0]._data = [[data['img_metas'][0]._data]]
    data['proposals'] = [DC([data['proposals']])]
    return data

def search_performance_cuhk(dataset, name_to_det_feat, result, idx, gallery_size=100):
    """
    gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
    gallery_feat (list of ndarray): n_det x D features per image
    probe_feat (list of ndarray): D dimensional features per probe image
    threshold (float): filter out gallery detections whose scores below this
    gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                        -1 for using full set
    dump_json (str): Path to save the results as a JSON file or None
    """
    use_full_set = gallery_size == -1
    fname = "TestG{}".format(gallery_size if not use_full_set else 50)
    protoc = loadmat(osp.join(dataset.root_dir, "annotation/test/train_test", fname + ".mat"))[fname].squeeze()

    topk = [1, 5, 10]
    # ret = {"image_root": dataset.data_path, "results": []}
    y_true, y_score = [], []
    imgs, rois = [], []
    count_gt, count_tp = 0, 0
    # Get L2-normalized feature vector
    feat_p = normalize(result[0][0][:, 5:5+256]).ravel()
    # Ignore the probe image
    probe_imname = str(protoc["Query"][idx]["imname"][0, 0][0])
    probe_roi = protoc["Query"][idx]["idlocate"][0, 0][0].astype(np.int32)
    probe_roi[2:] += probe_roi[:2]
    probe_gt = []
    tested = set([probe_imname])
    # 1. Go through the gallery samples defined by the protocol
    for item in protoc["Gallery"][idx].squeeze():
        gallery_imname = str(item[0][0])
        # some contain the probe (gt not empty), some not
        gt = item[1][0].astype(np.int32)
        count_gt += gt.size > 0
        # compute distance between probe and gallery dets
        if gallery_imname not in name_to_det_feat:
            continue
        det, feat_g = name_to_det_feat[gallery_imname]
        score = det[:, 4]

        # get L2-normalized feature matrix NxD
        assert feat_g.size == np.prod(feat_g.shape[:2])
        feat_g = feat_g.reshape(feat_g.shape[:2])

        feat_g = score[:, np.newaxis] * feat_g
        # compute cosine similarities
        sim = feat_g.dot(feat_p).ravel()
        # assign label for each det
        label = np.zeros(len(sim), dtype=np.int32)
        if gt.size > 0:
            w, h = gt[2], gt[3]
            gt[2:] += gt[:2]
            probe_gt.append({"img": str(gallery_imname), "roi": map(float, list(gt))})
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
        tested.add(gallery_imname)
    # 2. Go through the remaining gallery images if using full set
    if use_full_set:
        for gallery_imname in dataset.image_index:
            if gallery_imname in tested:
                continue
            if gallery_imname not in name_to_det_feat:
                continue
            det, feat_g = name_to_det_feat[gallery_imname]
            # get L2-normalized feature matrix NxD
            assert feat_g.size == np.prod(feat_g.shape[:2])
            feat_g = feat_g.reshape(feat_g.shape[:2])
            # compute cosine similarities
            sim = feat_g.dot(feat_p).ravel()
            # guaranteed no target probe in these gallery images
            label = np.zeros(len(sim), dtype=np.int32)
            y_true.extend(list(label))
            y_score.extend(list(sim))
            imgs.extend([gallery_imname] * len(sim))
            rois.extend(list(det))
    # 3. Compute AP for this probe (need to scale by recall rate)
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    assert count_tp <= count_gt
    recall_rate = count_tp * 1.0 / count_gt
    ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
    inds = np.argsort(y_score)[::-1]
    y_score = y_score[inds]
    y_true = y_true[inds]
    acc = ([min(1, sum(y_true[:k])) for k in topk])
    # 4. Save result for JSON dump
    new_entry = {
        "probe_img": str(probe_imname),
        "probe_roi": list(probe_roi.squeeze()),
        "probe_gt": probe_gt,
        'ap':ap,
        'acc':acc,
        'pid':-1,
        "gallery": [],
        "image_root": dataset.data_path
    }
    # only save top-10 predictions
    for k in range(10):
        new_entry["gallery"].append({
                "img": str(imgs[inds[k]]),
                "roi": list(rois[inds[k]]),
                "score": float(y_score[k]),
                "correct": int(y_true[k]),
            }
        )
    # ret["results"].append(new_entry)
    print(y_true[:5])
    return new_entry

def main():
    info = get_info_sota()
    dataset_name = 'CUHK'
    model = load_model(info[dataset_name])
    query_data_loader, name_to_det_feat, psdb_dataset = get_cuhk_dataset_info(info['CUHK'])
    
    # for idx in [12, 34, 5]:
    idx = 14
    data = get_cuhk_data(query_data_loader, idx)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    search_performance_cuhk(psdb_dataset, name_to_det_feat, result, idx, gallery_size=100)

if __name__ == '__main__':
    main()
