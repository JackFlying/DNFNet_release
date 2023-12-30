_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4_reid_norm_unsu.py',
    '../_base_/datasets/coco_reid_unsup_prw.py',
    '../_base_/schedules/schedule_1x_reid_norm_base.py', '../_base_/default_runtime.py'
]
TEST = False 
USE_PART_FEAT = False
GLOBAL_WEIGHT = 0.9 # not used
UNCERTAINTY = False  # 时候用dual label
USE_GFN = False
model = dict(
    roi_head=dict(
        type='ReidRoIHeadDNFNet',
        use_gfn=USE_GFN,
        use_RoI_Align_feat=False,
        use_part_feat=USE_PART_FEAT,
        scene_emb_size=56,
        shared_head=dict(
            type='ResLayer',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True),
        bbox_head=dict(
            type='DNFNetHead',
            id_num=18048,
            testing=TEST,
            rcnn_bbox_bn=False,
            cluster_top_percent=0.6,
            momentum=0.2,
            IoU_memory_clip=[0.2, 0.9],
            use_cluster_hard_loss=True,
            use_quaduplet_loss=True,
            use_part_feat=USE_PART_FEAT,
            num_features=256,
            cluster_mean_method='naive',    # ['naive', 'time_consistency', 'soft_time_consistency']
            tc_winsize=100, # for time_consistency
            decay_weight=-0.001, # for soft_time_consistency
            update_method='momentum',    # ['momentum', 'iou', 'max_iou', 'momentum_max_iou', 'gt']
        )
    )
)
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', 
        img_scale=[(667, 400),(1000, 600), (1333, 800), (1500,900), (1666, 1000), (2000, 1200)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1500, 900),
        # img_scale=(1666, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
query_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1500, 900),
        # img_scale=(1666, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['proposals']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1500, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline, query_test_pipeline=None),
    train_cluster=dict(pipeline=val_pipeline, query_test_pipeline=None),
    val=dict(pipeline=test_pipeline, query_test_pipeline=None),
    test=dict(pipeline=test_pipeline,
        query_test_pipeline=query_test_pipeline,
    ))
# optimizer_config = dict(_delete_=True, grad_clip=None)

# optimizer
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2242,
    warmup_ratio=1.0 / 200,
    step=[16, 22])
total_epochs = 30

SPCL=True
PSEUDO_LABELS = dict(
    freq=1, # epochs
    use_outliers=True,
    norm_feat=True,
    norm_center=True,
    SpCL=True,
    cluster='FINCH_context_SpCL_Plus',   # dbscan_context, FINCH_context, FINCH_context_SpCL, FINCH_context_SpCL_Plus
    eps=[0.68, 0.7, 0.72],
    # eps=[0.70],
    min_samples=4, # for dbscan
    dist_metric='jaccard',
    k1=30, # for jaccard distance
    k2=6, # for jaccard distance
    search_type=0, # 0,1,2 for GPU, 3 for CPU (work for faiss)
    cluster_num=None,
    iters=0,    # defalut: 3
    lambda_scene=0,
    lambda_person=0.1,
    context_method='max',   # defalut: 'zero'
    threshold=0.5,
    use_post_process=False,
    filter_threshold=0.2,
    use_crop=False,
    use_k_reciprocal_nearest=False,
    K=10,
    part_feat=dict(use_part_feat=USE_PART_FEAT, 
                    global_weight=GLOBAL_WEIGHT,
                    uncertainty=UNCERTAINTY,
                    uncertainty_threshold=0.5,
                    global_weights=[1.0, 0.9],
                    ),
    inter_cluster=dict(
                    use_inter_cluster=False, # USE_PART_FEAT==False启动
                    T=1,
                    )
)
workflow = [('train', 1)]
evaluation = dict(start=20, interval=2, metric='bbox')
testing = TEST
save_features = True
restart = False