import os.path as osp
import pickle
import shutil
import tempfile
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results, tensor2imgs

def single_gpu_test_cuhk(model,
                    data_loader,
                    query_data_loader,
                    load_gallery=False,
                    max_iou_with_proposal=False):
    model.eval()
    gboxes = []
    gfeatures = []
    pfeatures = []
    if not load_gallery:
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            # if i > 50:
            #     break
            with torch.no_grad():
                data['use_crop'] = False
                result = model(return_loss=False, rescale=True, **data)

            batch_size = len(result)
            if result[0][0].shape[0]>0:
                gboxes.append(result[0][0][:, :5])
                gfeatures.append(result[0][0][:, 5:])
            else:
                gboxes.append(np.zeros((0, 5), dtype=np.float32))
                gfeatures.append(np.zeros((0, 256), dtype=np.float32))
            
            for _ in range(batch_size):
                prog_bar.update()
    
    try:
        if isinstance(model.module.roi_head.bbox_head, nn.ModuleList):
            for i in range(len(model.module.roi_head.bbox_head)):
                model.module.roi_head.bbox_head[i].proposal_score_max=True
        else:
            model.module.roi_head.bbox_head.proposal_score_max=True
    except:
        print('set proposal max fail')
        exit()
        pass

    dataset = query_data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(query_data_loader):
        # if i > 50:
        #     break
        if max_iou_with_proposal:
            proposal = data.pop('proposals')[0][0][0].cpu().numpy()
        with torch.no_grad():
            data['use_crop'] = False
            result = model(return_loss=False, rescale=True, **data)
            proposal = data['proposals'][0][0][0].cpu().numpy()
            scales = data['img_metas'][0].data[0][0]['scale_factor']
        batch_size = len(result)
    
        if max_iou_with_proposal:
            scales = data['img_metas'][0].data[0][0]['scale_factor']
            proposal = proposal/scales

        if max_iou_with_proposal:
            boxes = result[0][0][:, :4]
            areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
            area = (proposal[2] - proposal[0] + 1) * (proposal[3] - proposal[1] + 1)
            xx1 = np.maximum(proposal[0], boxes[:, 0])
            yy1 = np.maximum(proposal[1], boxes[:, 1])
            xx2 = np.minimum(proposal[2], boxes[:, 2])
            yy2 = np.minimum(proposal[3], boxes[:, 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inters = w * h
            IoU = inters / (areas + area - inters)
            iou_i = np.argmax(IoU)
            pfeatures.append(result[0][0][iou_i:iou_i+1, 5:])
        else:
            pfeatures.append(result[0][0][:, 5:])

        for _ in range(batch_size):
            prog_bar.update()

    return [gboxes, gfeatures, pfeatures]


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in tqdm(enumerate(data_loader)):
        # print(data)
        data['use_crop'] = False
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # RoI_Align_feat = result[0][0][:, 5+256:]
            # RoI_Align_feat = RoI_Align_feat.reshape(-1, 2048, 7, 7)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
