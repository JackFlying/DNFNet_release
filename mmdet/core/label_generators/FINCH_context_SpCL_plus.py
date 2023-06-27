from email.policy import default
from symbol import global_stmt
import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings
import torch
import torch.nn.functional as F
import collections
from tqdm import tqdm

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

ANN_THRESHOLD = 70000

def clust_rank(mat, bottom_mat, top_mat, initial_rank=None, distance='cosine', cfg=None, intra_context_mask=None):  # cosine: 1 - cos
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = np.array([])
    elif s <= ANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        bottom_orig_dist = metrics.pairwise.pairwise_distances(bottom_mat, bottom_mat, metric=distance)
        top_orig_dist = metrics.pairwise.pairwise_distances(top_mat, top_mat, metric=distance)
        orig_dist = cfg.PSEUDO_LABELS.part_feat.global_weight * orig_dist + cfg.PSEUDO_LABELS.part_feat.part_weight * (bottom_orig_dist + top_orig_dist)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1) # 这里计算相似度越高就为0
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean_old(M, u):
    _, nf = np.unique(u, return_counts=True)
    idx = np.argsort(u)
    M = M[idx, :]
    M = np.vstack((np.zeros((1, M.shape[1])), M))

    np.cumsum(M, axis=0, out=M)
    cnf = np.cumsum(nf)
    nf1 = np.insert(cnf, 0, 0)
    nf1 = nf1[:-1]

    M = M[cnf, :] - M[nf1, :]
    M = M / nf[:, None]
    return M


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]

def get_merge(c, u, data, bottom_data, top_data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u
    mat = cool_mean(data, c)
    bmat = cool_mean(bottom_data, c)
    tmat = cool_mean(top_data, c)
    return c, mat, bmat, tmat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data_list, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True, cfg=None, intra_context_mask=None):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    # Cast input data to float32
    data, bottom_data, top_data = data_list
    data = np.array(data.cpu())
    bottom_data = np.array(bottom_data.cpu())
    top_data = np.array(top_data.cpu())
    intra_context_mask = np.array(intra_context_mask.cpu())
    
    if initial_rank is not None:
        initial_rank = np.array(initial_rank.cpu())
    
    data = data.astype(np.float32)
    bottom_data = bottom_data.astype(np.float32)
    top_data = top_data.astype(np.float32)

    min_sim = None
    adj, orig_dist = clust_rank(data, bottom_data, top_data, initial_rank, distance, cfg, intra_context_mask)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat, bmat, tmat = get_merge([], group, data, bottom_data, top_data) # 求聚类中心

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        adj, orig_dist = clust_rank(mat, bmat, tmat, initial_rank, distance, cfg)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat, bmat, tmat = get_merge(c_, u, data, bottom_data, top_data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c

def list_duplicates(seq):
    tally = collections.defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    dups = [(key,locs) for key,locs in tally.items() if len(locs)>1]    # key表示簇的label,locs表示簇中样本的index
    return dups

@torch.no_grad()
def process_label_with_intra_context(labels, centers, features, inds, num_classes):
    # if the persons in the same image are clustered in the same clust, remove it
    N_p = features.shape[0]
    N_c = centers.shape[0]
    assert num_classes == N_c
    assert N_p == labels.shape[0]
    assert N_p == inds.shape[0]
    unique_inds = set(inds.cpu().numpy())
    print("num_classes_pre", num_classes)
    for uid in unique_inds: # image idx
        if(uid == 0):
            continue
        b = (inds == uid)
        tmp_id = b.nonzero()
        tmp_labels = labels[tmp_id] # instance label belong to same image
        dups = list_duplicates(list(tmp_labels.squeeze(1).cpu().numpy()))
        if len(dups) > 0:
            for dup in dups:
                # dup[0]表示簇标签, dup[1]表示属于该簇同时属于同一张图片的样本
                tmp_center = centers[dup[0]].cpu().numpy()
                # TODO 重新求聚类中心，去除掉待筛选的样本
                # is_selected = (labels == dup[0])
                # is_selected[tmp_id[dup[1]].squeeze(1)] = False
                # tmp_center = features[is_selected].mean(0)
                
                tmp_features = features[tmp_id[dup[1]].squeeze(1)].cpu().numpy()
                sim = np.dot(tmp_center, tmp_features.transpose())
                idx = np.argmax(sim)
                for i in range(len(sim)):
                    if i != idx:
                        labels[tmp_id[dup[1][i]]] = num_classes
                        centers = torch.cat((centers, features[tmp_id[dup[1][i]]])) # 增加聚类中心，实际上为异常点的中心
                        num_classes += 1
    print("num_classes_after", num_classes)
    assert num_classes == centers.shape[0]
    return labels, centers, num_classes

@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    # import ipdb;    ipdb.set_trace()
    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    return centers

def get_img_sim_by_max(inst_sim_matrix, split_num):
    """
        获得图片之间的相似度
        1. 计算instance与图片之间的最大相似度
        2. 计算图片和图片之间的最大相似度
    """
    inst_sim_matrix_split = inst_sim_matrix.split(split_num, -1)
    img2inst_sim = []
    for i in range(len(split_num)):
        values, _ = torch.max(inst_sim_matrix_split[i], dim=-1)
        img2inst_sim.append(values)

    img2inst_sim = torch.stack(img2inst_sim, dim=0) # [img_len, inst_len]
    img2inst_sim_split = img2inst_sim.split(split_num, -1)

    img2img_sim = []
    for i in range(len(split_num)):
        values, _ = torch.max(img2inst_sim_split[i], -1)
        img2img_sim.append(values)
    img2img_sim = torch.stack(img2img_sim, dim=0)
    return img2img_sim

def get_img_sim_by_mean(inst_sim_matrix, split_num, threshold):
    """
        获得图片之间的相似度
        需要找到正样本的集合

        1.将相似度小于阈值的设置为0
        2.计算instance与image之间的相似度
        3.计算instance与instance之间的相似度
    """
    inst_sim_matrix = inst_sim_matrix * (inst_sim_matrix > threshold)
    inst_sim_matrix_split = inst_sim_matrix.split(split_num, -1)
    img2inst_sim = []
    for i in range(len(split_num)):
        values = torch.sum(inst_sim_matrix_split[i], dim=-1)
        img2inst_sim.append(values)

    img2inst_sim = torch.stack(img2inst_sim, dim=0) # [img_len, inst_len]
    img2inst_sim_split = img2inst_sim.split(split_num, -1)

    img2img_sim = []
    for i in range(len(split_num)):
        values = torch.sum(img2inst_sim_split[i], -1)
        img2img_sim.append(values)
    img2img_sim = torch.stack(img2img_sim, dim=0)
    return img2img_sim

def get_extend_sim_matrix(sim, split_num):
    inst2img_sim = []
    for i in range(len(split_num)):
        inst2img_sim.append(sim[:, i:i+1].repeat(1, split_num[i]))
    inst2img_sim = torch.cat(inst2img_sim, dim=-1)    # [img_len, inst_len]
    
    img2img_sim_extend = []
    for i in range(len(split_num)):
        img2img_sim_extend.append(inst2img_sim[i:i+1, :].repeat(split_num[i], 1))
    img2img_sim_extend = torch.cat(img2img_sim_extend, dim=0)
    return img2img_sim_extend

def get_hybrid_sim(inst_sim_matrix, split_num, person_sim, scene_sim, lambda_person, lambda_scene, intra_context_mask):
    """
        将上下文相似度加到视觉相似度上面
        1. 将上下文相似度维度进行扩充满足视觉相似度
        2. 计算最近邻
    """
    person_sim_extend = get_extend_sim_matrix(person_sim, split_num)
    scene_sim_extend = get_extend_sim_matrix(scene_sim, split_num)
    hybrid_sim_matrix = inst_sim_matrix + lambda_person * person_sim_extend + lambda_scene * scene_sim_extend
    # 将对角线位置填充为负无穷大,自身不能作为最近邻
    hybrid_sim_matrix_ = hybrid_sim_matrix.clone()
    hybrid_sim_matrix_ = hybrid_sim_matrix_.fill_diagonal_(-100.)
    # hybrid_sim_matrix_[intra_context_mask == False] = -100
    _, initial_rank = torch.max(hybrid_sim_matrix_, dim=-1)
    return hybrid_sim_matrix, initial_rank

@torch.no_grad()
def label_generator_FINCH_context_single(cfg, iters, features_list, initial_rank, all_inds, intra_context_mask):
    features = features_list[0].cpu()
    labels_set, num_classes_set, req_c = FINCH(features_list, initial_rank=initial_rank, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True, cfg=cfg, intra_context_mask=intra_context_mask)
    labels = torch.from_numpy(labels_set[:, iters]).long()
    num_classes = num_classes_set[iters]
    centers = generate_cluster_features(labels.tolist(), features)
    labels, _, num_classes = process_label_with_intra_context(labels, centers, features, all_inds, num_classes)
    return labels, _, num_classes

@torch.no_grad()
def label_generator_FINCH_context_SpCL(cfg, features_list, initial_rank, cuda=True, indep_thres=None, all_inds=None, intra_context_mask=None, **kwargs):
    iters = kwargs.get("iters", [0, 1, 2])
    features = features_list[0].cpu()
    labels_tight, _, num_classes_tight = label_generator_FINCH_context_single(cfg, iters[2], features_list, initial_rank, all_inds, intra_context_mask)
    labels_normal, _, num_classes = label_generator_FINCH_context_single(cfg, iters[1], features_list, initial_rank, all_inds, intra_context_mask)
    labels_loose, _, num_classes_loose = label_generator_FINCH_context_single(cfg, iters[0], features_list, initial_rank, all_inds, intra_context_mask)

    # compute R_indep and R_comp
    N = labels_normal.size(0)
    # 获得N*N的01矩阵,每一行相同的label置为1
    label_sim = (labels_normal.expand(N, N).eq(labels_normal.expand(N, N).t()).float())
    label_sim_tight = (labels_tight.expand(N, N).eq(labels_tight.expand(N, N).t()).float())
    label_sim_loose = (labels_loose.expand(N, N).eq(labels_loose.expand(N, N).t()).float())
    # 计算score
    R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)  # [N]
    R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1) # [N]
    assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
    assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

    cluster_R_comp, cluster_R_indep = (
        collections.defaultdict(list),
        collections.defaultdict(list),
    )
    # 统计簇的score
    cluster_img_num = collections.defaultdict(int)
    for comp, indep, label in zip(R_comp, R_indep, labels_normal):
        cluster_R_comp[label.item()].append(comp.item())
        cluster_R_indep[label.item()].append(indep.item())
        cluster_img_num[label.item()] += 1
    # 统计instance的最小score最为簇的score
    cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
    cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
    # 对于大于1的簇, 计算indep指标
    cluster_R_indep_noins = [
        iou
        for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
        if cluster_img_num[num] > 1
    ]
    # 根据indep score计算indep阈值, 对score排序,选择90%位置作为阈值
    if indep_thres is None:
        indep_thres = np.sort(cluster_R_indep_noins)[
            min(
                len(cluster_R_indep_noins) - 1,
                np.round(len(cluster_R_indep_noins) * 0.9).astype("int"),
            )
        ]
    # 计算簇的大小
    labels_num = collections.defaultdict(int)
    for label in labels_normal:
        labels_num[label.item()] += 1

    centers = collections.defaultdict(list)
    outliers = 0

    print(len(cluster_R_indep), num_classes)
    
    for i, label in enumerate(labels_normal):
        label = label.item()
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

@torch.no_grad()
def label_generator_FINCH_context_SpCL_Plus(cfg, features, cuda=True, indep_thres=None, all_inds=None, **kwargs):

    unique_inds = set(all_inds.cpu().numpy())
    split_num = torch.zeros(len(unique_inds)).long()
    for i in range(all_inds.shape[0]):
        split_num[all_inds[i]] += 1
    split_num = split_num.tolist()
    
    instance_sim = features.mm(features.t())
    bottom_features = features
    top_features = features
    if cfg.PSEUDO_LABELS.part_feat.use_part_feat:
        print("-------------------------part based clustering---------------------------------")
        # hybrid_feature = cfg.PSEUDO_LABELS.part_feat.global_weight * features + cfg.PSEUDO_LABELS.part_feat.part_weight * (top_features + bottom_features)
        # hybrid_feature = F.normalize(hybrid_feature)
        # instance_sim = hybrid_feature.mm(hybrid_feature.t())
        bottom_features = torch.load("./saved_file/bottom_features.pth")
        top_features = torch.load("./saved_file/top_features.pth")
        bottom_feat_sim = bottom_features.mm(bottom_features.t())
        top_feat_sim = top_features.mm(top_features.t())
        part_feat_sim = bottom_feat_sim + top_feat_sim
        instance_sim = cfg.PSEUDO_LABELS.part_feat.global_weight * instance_sim + cfg.PSEUDO_LABELS.part_feat.part_weight * part_feat_sim

    person_sim = torch.zeros(len(unique_inds), len(unique_inds))
    if cfg.PSEUDO_LABELS.context_method == "max":
        img_sim = get_img_sim_by_max(instance_sim, split_num)
    elif cfg.PSEUDO_LABELS.context_method == "mean":
        img_sim = get_img_sim_by_mean(instance_sim, split_num, cfg.PSEUDO_LABELS.threshold)
    elif cfg.PSEUDO_LABELS.context_method == "zero":
        img_sim = torch.zeros(len(unique_inds), len(unique_inds))
    elif cfg.PSEUDO_LABELS.context_method == "scene":
        scene_features = torch.load("./saved_file/scene_features.pth")
        scene_sim = scene_features.mm(scene_features.t())

    # jaccard_coeff = 0.1
    # jaccard_dist = re_ranking_for_instance(features, k1=100)
    # instance_sim = (1 - jaccard_coeff) * instance_sim + jaccard_coeff * jaccard_dist
    
    # import ipdb;    ipdb.set_trace()
    # scene_sim = img_sim
    # if cfg.PSEUDO_LABELS.context_clip:
    #     scene_sim = scene_sim * (scene_sim > cfg.PSEUDO_LABELS.threshold)
    intra_context_mask = get_intra_context_mask(all_inds)
    hybrid_instance_sim, initial_rank = get_hybrid_sim(instance_sim, split_num, person_sim, img_sim, 0., cfg.PSEUDO_LABELS.lambda_scene, intra_context_mask)
    # if cfg.PSEUDO_LABELS.part_feat.use_part_feat:
    #     _, initial_rank_bottom = get_hybrid_sim(bottom_feat_sim, split_num, person_sim, img_sim, 0., cfg.PSEUDO_LABELS.lambda_scene)
    #     _, initial_rank_top = get_hybrid_sim(top_feat_sim, split_num, person_sim, img_sim, 0., cfg.PSEUDO_LABELS.lambda_scene)
    blabels = None
    tlabels = None
    if cfg.PSEUDO_LABELS.SpCL:
        labels, centers, num_classes, indep_thres = label_generator_FINCH_context_SpCL(cfg, [features, bottom_features, top_features], initial_rank, cuda, indep_thres, all_inds, intra_context_mask, **kwargs)
        # if cfg.PSEUDO_LABELS.part_feat.use_part_feat:
        #     blabels, centers, num_classes, indep_thres = label_generator_FINCH_context_SpCL(cfg, bottom_features, initial_rank_bottom, cuda, indep_thres, all_inds, **kwargs)
        #     tlabels, centers, num_classes, indep_thres = label_generator_FINCH_context_SpCL(cfg, top_features, initial_rank_top, cuda, indep_thres, all_inds, **kwargs)
    else:
        labels, centers, num_classes = label_generator_FINCH_context_single(cfg, 0, [features, bottom_features, top_features], initial_rank, all_inds, intra_context_mask)

    for i in range(cfg.PSEUDO_LABELS.iters):
        print("clustering iteration: {}".format(i + 1))
        unique_labels = set(labels.cpu().numpy())
        person_sim = torch.zeros(len(unique_inds), len(unique_inds))
        for label in unique_labels:
            b = (labels == label)
            tmp_id = b.nonzero()
            img_ids = all_inds[tmp_id]
            if len(img_ids) > 1:
                for i in range(len(img_ids)):
                    for j in range(0, i, 1):
                        person_sim[img_ids[i].item()][img_ids[j].item()] += instance_sim[tmp_id[i].item()][tmp_id[j].item()]
                        person_sim[img_ids[j].item()][img_ids[i].item()] += instance_sim[tmp_id[j].item()][tmp_id[i].item()]
        hybrid_instance_sim, initial_rank = get_hybrid_sim(instance_sim, split_num, person_sim, img_sim, cfg.PSEUDO_LABELS.lambda_person, cfg.PSEUDO_LABELS.lambda_scene, intra_context_mask)
        # if cfg.PSEUDO_LABELS.part_feat.use_part_feat:
        #     _, initial_rank_bottom = get_hybrid_sim(bottom_feat_sim, split_num, person_sim, img_sim, cfg.PSEUDO_LABELS.lambda_person, cfg.PSEUDO_LABELS.lambda_scene)
        #     _, initial_rank_top = get_hybrid_sim(top_feat_sim, split_num, person_sim, img_sim, cfg.PSEUDO_LABELS.lambda_person, cfg.PSEUDO_LABELS.lambda_scene)
        if cfg.PSEUDO_LABELS.SpCL:
            labels, centers, num_classes, indep_thres = label_generator_FINCH_context_SpCL(cfg, [features, bottom_features, top_features], initial_rank, cuda, indep_thres, all_inds, intra_context_mask, **kwargs)
            # if cfg.PSEUDO_LABELS.part_feat.use_part_feat:
            #     blabels, centers, num_classes, indep_thres = label_generator_FINCH_context_SpCL(cfg, bottom_features, initial_rank_bottom, cuda, indep_thres, all_inds, **kwargs)
            #     tlabels, centers, num_classes, indep_thres = label_generator_FINCH_context_SpCL(cfg, top_features, initial_rank_top, cuda, indep_thres, all_inds, **kwargs)
        else:
            labels, centers, num_classes = label_generator_FINCH_context_single(cfg, 0, [features, bottom_features, top_features], initial_rank, all_inds, intra_context_mask)

    return labels, centers, num_classes, indep_thres, blabels, tlabels

@torch.no_grad()
def get_intra_context_mask(inds):
    # 获取mask矩阵，将属于同一张图片的行人的相似度设置为无穷大
    N = inds.shape[0]
    intra_context_mask = torch.ones((N, N)).bool()
    unique_inds = set(inds.cpu().numpy())
    for uid in unique_inds: # image idx
        b = (inds == uid)
        tmp_id = b.nonzero().squeeze(-1).tolist()
        for i in range(len(tmp_id)):
            for j in range(0, i, 1):
                intra_context_mask[tmp_id[i]][tmp_id[j]] = intra_context_mask[tmp_id[j]][tmp_id[i]] = False
    return intra_context_mask

def pairwiseDis(qFeature, gFeature):  # 246s
    # 计算余弦距离,数值越大,相似度越低
    x, y = F.normalize(qFeature), F.normalize(gFeature)
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    return disMat.clamp_(min=1e-5)

@torch.no_grad()
def re_ranking_for_instance(memory_features, k1, k2=6):
    """
        求扩展K互最近邻
        rank_k_matrix: [N, N]
    """
    N = memory_features.shape[0]
    original_dist = pairwiseDis(memory_features, memory_features).cpu().numpy()
    original_dist = np.transpose(original_dist / (np.max(original_dist, axis=0)))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)
    for i in range(N):
        forward_k_neigh_index = initial_rank[i,:k1+1] 
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
        
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=N

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(N):
        temp_min = np.zeros(shape=[1, N], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    
    jaccard_dist = 1 - jaccard_dist
    return jaccard_dist