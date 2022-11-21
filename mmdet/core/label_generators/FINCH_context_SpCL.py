from concurrent.futures import thread
import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings
import torch
import torch.nn.functional as F
import collections

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

ANN_THRESHOLD = 70000

def clust_rank(mat, initial_rank=None, distance='cosine'):  # cosine: 1 - cos
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = np.array([])
    elif s <= ANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
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


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


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


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True):
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
    data = np.array(data.cpu())
    if initial_rank is not None:
        initial_rank = np.array(initial_rank.cpu())
    
    data = data.astype(np.float32)

    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 3
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        adj, orig_dist = clust_rank(mat, initial_rank, distance)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

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
    for uid in unique_inds:
        b = inds == uid
        tmp_id = b.nonzero()
        tmp_labels = labels[tmp_id] # instance belong to same image
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
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
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

def get_img_sim_by_max2(inst_sim_matrix, split_num, threshold):
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
    img2img_sim = img2img_sim * (img2img_sim > threshold)
    # torch.save(img2img_sim, "./img2img_sim.pth")
    return img2img_sim

def get_img_sim_by_sum(inst_sim_matrix, split_num, threshold):
    """
        所以匹配上的pair的相似度的和,取均值
        1.将相似度小于阈值的设置为0
        2.计算instance与image之间的相似度
        3.计算instance与instance之间的相似度
    """
    mask = inst_sim_matrix > threshold
    inst_sim_matrix = inst_sim_matrix * mask
    inst_sim_matrix_split = inst_sim_matrix.split(split_num, -1)
    mask_split = mask.split(split_num, -1)
    img2inst_sim = []
    mask_sum = []
    for i in range(len(split_num)):
        values = torch.sum(inst_sim_matrix_split[i], dim=-1)
        mask_sum.append(mask_split[i].sum(-1))
        img2inst_sim.append(values)

    img2inst_sim = torch.stack(img2inst_sim, dim=0) # [img_len, inst_len]
    mask_sum = torch.stack(mask_sum, dim=0) # [img_len, inst_len]
    mask_split = mask_sum.split(split_num, -1)
    img2inst_sim_split = img2inst_sim.split(split_num, -1)

    img2img_sim = []
    mask_sum = []
    for i in range(len(split_num)):
        values = torch.sum(img2inst_sim_split[i], -1)
        img2img_sim.append(values)
        mask_sum.append(mask_split[i].sum(-1))
    mask_sum = torch.stack(mask_sum, dim=0)
    mask_sum_select = torch.where(mask_sum == 0, torch.tensor(1), mask_sum)
    img2img_sim = torch.stack(img2img_sim, dim=0) / mask_sum_select
    return img2img_sim

def get_img_sim_by_img():
    img_feats = torch.load("./img_feats.pth", map_location='cpu')
    img_feats = F.normalize(img_feats)
    img2img_sim = img_feats.mm(img_feats.t())
    return img2img_sim

def get_hybrid_sim(inst_sim_matrix, img_sim, split_num, lambda_sim):
    """
        将上下文相似度加到视觉相似度上面
        1. 将上下文相似度维度进行扩充满足视觉相似度
        2. 计算最近邻
    """
    inst2img_sim = []
    for i in range(len(split_num)):
        inst2img_sim.append(img_sim[:, i:i+1].repeat(1, split_num[i]))
    inst2img_sim = torch.cat(inst2img_sim, dim=-1)    # [img_len, inst_len]
    
    img2img_sim_extend = []
    for i in range(len(split_num)):
        img2img_sim_extend.append(inst2img_sim[i:i+1, :].repeat(split_num[i], 1))
    img2img_sim_extend = torch.cat(img2img_sim_extend, dim=0)
    
    hybrid_sim_matrix = inst_sim_matrix + lambda_sim * img2img_sim_extend
    # 将对角线位置填充为负无穷大,自身不能作为最近邻
    hybrid_sim_matrix.fill_diagonal_(-100.)
    _, initial_rank = torch.max(hybrid_sim_matrix, dim=-1)
    return hybrid_sim_matrix, initial_rank

@torch.no_grad()
def label_generator_FINCH_context_single(cfg, iters, features, all_inds=None):

    unique_inds = set(all_inds.cpu().numpy())
    split_num = torch.zeros(len(unique_inds)).long()
    for i in range(all_inds.shape[0]):
        split_num[all_inds[i]] += 1
    split_num = split_num.tolist()

    similarity_matrix = features.mm(features.t())
    if cfg.PSEUDO_LABELS.context_method == "max":
        img_sim = get_img_sim_by_max(similarity_matrix, split_num)
    elif cfg.PSEUDO_LABELS.context_method == "max+":
        img_sim = get_img_sim_by_max2(similarity_matrix, split_num, cfg.PSEUDO_LABELS.threshold)
    elif cfg.PSEUDO_LABELS.context_method == "sum":
        img_sim = get_img_sim_by_sum(similarity_matrix, split_num, cfg.PSEUDO_LABELS.threshold)
    elif cfg.PSEUDO_LABELS.context_method == "img":
        img_sim = get_img_sim_by_img()

    hybrid_sim_matrix, initial_rank = get_hybrid_sim(similarity_matrix, img_sim, split_num, cfg.PSEUDO_LABELS.LAMBDA_SIM)
    labels_set, num_classes_set, req_c = FINCH(features, initial_rank=initial_rank, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True)
    labels = torch.from_numpy(labels_set[:, iters]).long()
    num_classes = num_classes_set[iters]
    centers = generate_cluster_features(torch.unique(labels), features)
    labels, _, num_classes = process_label_with_context(labels, centers, features, all_inds, num_classes)
    return labels, centers, num_classes

@torch.no_grad()
def label_generator_FINCH_context_SpCL(cfg, features, cuda=True, indep_thres=None, all_inds=None, **kwargs):

    features = features.cpu()

    labels_tight, centers_tight, num_classes_tight = label_generator_FINCH_context_single(cfg, 2, features, all_inds=all_inds)
    labels_normal, centers_normal, num_classes = label_generator_FINCH_context_single(cfg, 1, features, all_inds=all_inds)
    labels_loose, centers_loose, num_classes_loose = label_generator_FINCH_context_single(cfg, 0, features, all_inds=all_inds)

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
def label_generator_FINCH_context(cfg, features, cuda=True, indep_thres=None, all_inds=None, **kwargs):

    unique_inds = set(all_inds.cpu().numpy())
    split_num = torch.zeros(len(unique_inds)).long()
    for i in range(all_inds.shape[0]):
        split_num[all_inds[i]] += 1
    split_num = split_num.tolist()

    similarity_matrix = features.mm(features.t())
    if cfg.PSEUDO_LABELS.context_method == "max":
        img_sim = get_img_sim_by_max(similarity_matrix, split_num)
    elif cfg.PSEUDO_LABELS.context_method == "max+":
        img_sim = get_img_sim_by_max2(similarity_matrix, split_num, cfg.PSEUDO_LABELS.threshold)
    elif cfg.PSEUDO_LABELS.context_method == "sum":
        img_sim = get_img_sim_by_sum(similarity_matrix, split_num, cfg.PSEUDO_LABELS.threshold)
    elif cfg.PSEUDO_LABELS.context_method == "img":
        img_sim = get_img_sim_by_img()
    
    hybrid_sim_matrix, initial_rank = get_hybrid_sim(similarity_matrix, img_sim, split_num, cfg.PSEUDO_LABELS.LAMBDA_SIM)
    labels_set, num_classes_set, req_c = FINCH(features, initial_rank=initial_rank, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True)
    labels = torch.from_numpy(labels_set[:, cfg.PSEUDO_LABELS.iters]).long()
    num_classes = num_classes_set[cfg.PSEUDO_LABELS.iters]
    centers = generate_cluster_features(torch.unique(labels), features)
    labels, _, num_classes = process_label_with_context(labels, centers, features, all_inds, num_classes)
    return labels, centers, num_classes, indep_thres