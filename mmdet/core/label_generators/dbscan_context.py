# Written by Yixiao Ge

import collections

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from .compute_dist import build_dist

__all__ = ["label_generator_dbscan_context_single", "label_generator_context_dbscan"]

def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

@torch.no_grad()
def label_generator_dbscan_context_single(cfg, features, dist, eps, **kwargs):
    assert isinstance(dist, np.ndarray)

    # clustering
    min_samples = 4
    use_outliers = True

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1,)
    labels = cluster.fit_predict(dist)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)    # >1的cluster数量

    # cluster labels -> pseudo labels
    # compute cluster centers
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_clusters + outliers
            outliers += 1

        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters

def list_duplicates(seq):
    tally = collections.defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    dups = [(key,locs) for key,locs in tally.items() if len(locs)>1]
    return dups


@torch.no_grad()
def process_label_with_context(labels, centers, features, inds, num_classes):
    # if the persons in the same image are clustered in the same clust, remove it
    N_p = features.shape[0]
    N_c = centers.shape[0]
    assert num_classes == N_c
    assert N_p == labels.shape[0]
    assert N_p == inds.shape[0]
    unique_inds = set(inds.cpu().numpy())
    #print(unique_inds)
    #print(inds)
    for uid in unique_inds:
        #print("uid", uid)
        b = inds == uid
        tmp_id = b.nonzero()
        #print("tmp_id", tmp_id)
        tmp_labels = labels[tmp_id]
        #print(tmp_labels.squeeze(1), tmp_labels.squeeze(1).shape, list(tmp_labels.squeeze(1).cpu().numpy()))
        dups = list_duplicates(list(tmp_labels.squeeze(1).cpu().numpy()))
        if len(dups) > 0:
            for dup in dups:
                #print(features.shape, centers.shape)
                tmp_center = centers[dup[0]].cpu().numpy()
                #print(tmp_center.shape)
                tmp_features = features[tmp_id[dup[1]].squeeze(1)].cpu().numpy()
                #print(tmp_id[dup[1]].squeeze(1), tmp_features.shape)
                sim = np.dot(tmp_center, tmp_features.transpose())
                #print(sim)
                idx = np.argmax(sim)
                for i in range(len(sim)):
                    if i != idx:
                        labels[tmp_id[dup[1][i]]] = num_classes
                        centers = torch.cat((centers, features[tmp_id[dup[1][i]]]))
                        num_classes += 1
                        #print(centers.shape, num_classes)
    assert num_classes == centers.shape[0]
    return labels, centers, num_classes

@torch.no_grad()
def label_generator_dbscan_context(cfg, features, cuda=True, indep_thres=None, all_inds=None, **kwargs):
    assert cfg.PSEUDO_LABELS.cluster == "dbscan_context"

    if not cuda:
        cfg.PSEUDO_LABELS.search_type = 3

    # # compute distance matrix by features
    dist = build_dist(cfg.PSEUDO_LABELS, features, verbose=True)

    features = features.cpu()

    # clustering
    eps = cfg.PSEUDO_LABELS.eps

    if len(eps) == 1:
        # normal clustering
        labels, centers, num_classes = label_generator_dbscan_context_single(
            cfg, features, dist, eps[0]
        )
        if all_inds is not None:
            labels, centers, num_classes = process_label_with_context(labels, centers, features, all_inds, num_classes)
        return labels, centers, num_classes, indep_thres

    else:
        assert (
            len(eps) == 3
        ), "three eps values are required for the clustering reliability criterion"

        print("adopt the reliability criterion for filtering clusters")
        eps = sorted(eps)
        labels_tight, centers_tight, num_classes_tight = label_generator_dbscan_context_single(cfg, features, dist, eps[0])
        labels_normal, centers_normal, num_classes = label_generator_dbscan_context_single(cfg, features, dist, eps[1])
        labels_loose, centers_loose, num_classes_loose = label_generator_dbscan_context_single(cfg, features, dist, eps[2])

        labels_tight, _, num_classes_tight = process_label_with_context(labels_tight, centers_tight, features, all_inds, num_classes_tight)
        labels_normal, _, num_classes = process_label_with_context(labels_normal, centers_normal, features, all_inds, num_classes)
        labels_loose, _, num_classes_loose = process_label_with_context(labels_loose, centers_loose, features, all_inds, num_classes_loose)

        # compute R_indep and R_comp
        N = labels_normal.size(0)
        label_sim = (labels_normal.expand(N, N).eq(labels_normal.expand(N, N).t()).float())
        label_sim_tight = (labels_tight.expand(N, N).eq(labels_tight.expand(N, N).t()).float())
        label_sim_loose = (labels_loose.expand(N, N).eq(labels_loose.expand(N, N).t()).float())

        R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
        R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
        assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
        assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

        cluster_R_comp, cluster_R_indep = (
            collections.defaultdict(list),
            collections.defaultdict(list),
        )
        cluster_img_num = collections.defaultdict(int)
        for comp, indep, label in zip(R_comp, R_indep, labels_normal):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()] += 1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [
            min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())
        ]
        cluster_R_indep_noins = [
            iou
            for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
            if cluster_img_num[num] > 1
        ]
        if indep_thres is None:
            indep_thres = np.sort(cluster_R_indep_noins)[
                min(
                    len(cluster_R_indep_noins) - 1,
                    np.round(len(cluster_R_indep_noins) * 0.9).astype("int"),
                )
            ]

        labels_num = collections.defaultdict(int)
        for label in labels_normal:
            labels_num[label.item()] += 1

        centers = collections.defaultdict(list)
        outliers = 0
        #print(cluster_R_indep)
        print(len(cluster_R_indep), num_classes)
        
        for i, label in enumerate(labels_normal):
            label = label.item()
            #print(label)
            indep_score = cluster_R_indep[label]
            comp_score = R_comp[i]
            if label == -1:
                assert not cfg.PSEUDO_LABELS.use_outliers, "exists a bug"
                continue
            if (indep_score > indep_thres) or (
                comp_score.item() > cluster_R_comp[label]
            ):
                if labels_num[label] > 1:
                    labels_normal[i] = num_classes + outliers
                    outliers += 1
                    labels_num[label] -= 1
                    labels_num[labels_normal[i].item()] += 1

            centers[labels_normal[i].item()].append(features[i])

        num_classes += outliers
        assert len(centers.keys()) == num_classes

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]
        centers = torch.stack(centers, dim=0)

        return labels_normal, centers, num_classes, indep_thres


# @torch.no_grad()
# def label_generator_FINCH_context_SpCL_Plus(cfg, features, cuda=True, indep_thres=None, all_inds=None, **kwargs):

#     unique_inds = set(all_inds.cpu().numpy())
#     split_num = torch.zeros(len(unique_inds)).long()
#     for i in range(all_inds.shape[0]):
#         split_num[all_inds[i]] += 1
#     split_num = split_num.tolist()
    
#     instance_sim = features.mm(features.t())

#     person_sim = torch.zeros(len(unique_inds), len(unique_inds))
#     if cfg.PSEUDO_LABELS.context_method == "max":
#         img_sim = get_img_sim_by_max(instance_sim, split_num)
#     elif cfg.PSEUDO_LABELS.context_method == "mean":
#         img_sim = get_img_sim_by_mean(instance_sim, split_num, cfg.PSEUDO_LABELS.threshold)
#     elif cfg.PSEUDO_LABELS.context_method == "zero":
#         img_sim = torch.zeros(len(unique_inds), len(unique_inds))
#     elif cfg.PSEUDO_LABELS.context_method == "scene":
#         scene_features = torch.load("./saved_file/scene_features.pth")
#         scene_sim = scene_features.mm(scene_features.t())

#     intra_context_mask = get_intra_context_mask(all_inds)
    
#     hybrid_instance_sim = build_dist(cfg.PSEUDO_LABELS, features, verbose=True)
    
#     hybrid_instance_sim, initial_rank = get_hybrid_sim(instance_sim, split_num, person_sim, img_sim, 0., cfg.PSEUDO_LABELS.lambda_scene, intra_context_mask)

#     labels, centers, num_classes, indep_thres = label_generator_dbscan_context(cfg, features, cuda, indep_thres, all_inds, **kwargs)

#     for i in range(cfg.PSEUDO_LABELS.iters):
#         print("clustering iteration: {}".format(i + 1))
#         unique_labels = set(labels.cpu().numpy())
#         person_sim = torch.zeros(len(unique_inds), len(unique_inds))
#         for label in unique_labels:
#             b = (labels == label)
#             tmp_id = b.nonzero()
#             img_ids = all_inds[tmp_id]
#             if len(img_ids) > 1:
#                 for i in range(len(img_ids)):
#                     for j in range(0, i, 1):
#                         person_sim[img_ids[i].item()][img_ids[j].item()] += instance_sim[tmp_id[i].item()][tmp_id[j].item()]
#                         person_sim[img_ids[j].item()][img_ids[i].item()] += instance_sim[tmp_id[j].item()][tmp_id[i].item()]
                        
#         hybrid_instance_sim, initial_rank = get_hybrid_sim(instance_sim, split_num, person_sim, img_sim, cfg.PSEUDO_LABELS.lambda_person, cfg.PSEUDO_LABELS.lambda_scene, intra_context_mask)
#         labels, centers, num_classes, indep_thres = label_generator_dbscan_context(cfg, features, cuda, indep_thres, all_inds, **kwargs)

#     return labels, centers, num_classes, indep_thres


# @torch.no_grad()
# def get_intra_context_mask(inds):
#     # 获取mask矩阵，将属于同一张图片的行人的相似度设置为无穷大
#     N = inds.shape[0]
#     intra_context_mask = torch.ones((N, N)).bool()
#     unique_inds = set(inds.cpu().numpy())
#     for uid in unique_inds: # image idx
#         b = (inds == uid)
#         tmp_id = b.nonzero().squeeze(-1).tolist()
#         for i in range(len(tmp_id)):
#             for j in range(0, i, 1):
#                 intra_context_mask[tmp_id[i]][tmp_id[j]] = intra_context_mask[tmp_id[j]][tmp_id[i]] = False
#     return intra_context_mask

# def get_hybrid_sim(inst_sim_matrix, split_num, person_sim, scene_sim, lambda_person, lambda_scene, intra_context_mask):
#     """
#         将上下文相似度加到视觉相似度上面
#         1. 将上下文相似度维度进行扩充满足视觉相似度
#         2. 计算最近邻
#     """
#     person_sim_extend = get_extend_sim_matrix(person_sim, split_num)
#     scene_sim_extend = get_extend_sim_matrix(scene_sim, split_num)
#     hybrid_sim_matrix = inst_sim_matrix + lambda_person * person_sim_extend + lambda_scene * scene_sim_extend
#     # 将对角线位置填充为负无穷大,自身不能作为最近邻
#     hybrid_sim_matrix_ = hybrid_sim_matrix.clone()
#     hybrid_sim_matrix_ = hybrid_sim_matrix_.fill_diagonal_(-100.)
#     hybrid_sim_matrix_[intra_context_mask == False] = -100
#     _, initial_rank = torch.max(hybrid_sim_matrix_, dim=-1)
#     return hybrid_sim_matrix, initial_rank


# def get_img_sim_by_max(inst_sim_matrix, split_num):
#     """
#         获得图片之间的相似度
#         1. 计算instance与图片之间的最大相似度
#         2. 计算图片和图片之间的最大相似度
#     """
#     inst_sim_matrix_split = inst_sim_matrix.split(split_num, -1)
#     img2inst_sim = []
#     for i in range(len(split_num)):
#         values, _ = torch.max(inst_sim_matrix_split[i], dim=-1)
#         img2inst_sim.append(values)

#     img2inst_sim = torch.stack(img2inst_sim, dim=0) # [img_len, inst_len]
#     img2inst_sim_split = img2inst_sim.split(split_num, -1)

#     img2img_sim = []
#     for i in range(len(split_num)):
#         values, _ = torch.max(img2inst_sim_split[i], -1)
#         img2img_sim.append(values)
#     img2img_sim = torch.stack(img2img_sim, dim=0)
#     return img2img_sim

# def get_img_sim_by_mean(inst_sim_matrix, split_num, threshold):
#     """
#         获得图片之间的相似度
#         需要找到正样本的集合

#         1.将相似度小于阈值的设置为0
#         2.计算instance与image之间的相似度
#         3.计算instance与instance之间的相似度
#     """
#     inst_sim_matrix = inst_sim_matrix * (inst_sim_matrix > threshold)
#     inst_sim_matrix_split = inst_sim_matrix.split(split_num, -1)
#     img2inst_sim = []
#     for i in range(len(split_num)):
#         values = torch.sum(inst_sim_matrix_split[i], dim=-1)
#         img2inst_sim.append(values)

#     img2inst_sim = torch.stack(img2inst_sim, dim=0) # [img_len, inst_len]
#     img2inst_sim_split = img2inst_sim.split(split_num, -1)

#     img2img_sim = []
#     for i in range(len(split_num)):
#         values = torch.sum(img2inst_sim_split[i], -1)
#         img2img_sim.append(values)
#     img2img_sim = torch.stack(img2img_sim, dim=0)
#     return img2img_sim

# def get_extend_sim_matrix(sim, split_num):
#     inst2img_sim = []
#     for i in range(len(split_num)):
#         inst2img_sim.append(sim[:, i:i+1].repeat(1, split_num[i]))
#     inst2img_sim = torch.cat(inst2img_sim, dim=-1)    # [img_len, inst_len]
    
#     img2img_sim_extend = []
#     for i in range(len(split_num)):
#         img2img_sim_extend.append(inst2img_sim[i:i+1, :].repeat(split_num[i], 1))
#     img2img_sim_extend = torch.cat(img2img_sim_extend, dim=0)
#     return img2img_sim_extend