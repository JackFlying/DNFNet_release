import sys

sys.path.append("./build")
import min_search_cuda
import numpy as np
import torch

n_query = 32
n_memory = 16384
all_ids = np.array(list(range(768)), dtype=np.int32)
while True:

    score_mat = np.random.rand(n_query, n_memory)
    cluster_ids = np.random.choice(all_ids, n_memory)
    query_ids = np.random.choice(all_ids, n_query)
    cluster_id_set = set(cluster_ids)
    cluster_ids_t = torch.tensor(cluster_ids)
    cluster_idxs = []
    for cid in cluster_id_set:
        c_idxs = (cluster_ids_t == cid).nonzero().view(-1).tolist()
        cluster_idxs.append(c_idxs)
    midxs = min_search_cuda.min_negative_search(score_mat, cluster_idxs)
#print(midxs)
