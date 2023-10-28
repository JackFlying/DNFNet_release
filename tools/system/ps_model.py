import torch
import torch.nn as nn
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

def load_model(info, gpu_id=None):
    cfg = Config.fromfile(info["config"])
    checkpoint_path = info["checkpoint"]
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        pass
        # model.CLASSES = dataset.CLASSES
    if gpu_id is None:
        model = MMDataParallel(model, device_ids=[torch.cuda.current_device()])
    else:
        model = MMDataParallel(model, device_ids=[gpu_id])
    model.eval()
    try:
        if isinstance(model.module.roi_head.bbox_head, nn.ModuleList):
            for i in range(len(model.module.roi_head.bbox_head)):
                model.module.roi_head.bbox_head[i].proposal_score_max=True
        else:
            model.module.roi_head.bbox_head.proposal_score_max=True
            model.module.roi_head.test_cfg.nms.iou_threshold=2
            model.module.roi_head.test_cfg.max_per_img=1000
    except:
        assert False, "setting fault"

    return model