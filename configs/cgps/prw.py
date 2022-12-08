_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4_reid_norm_unsu.py',
    '../_base_/datasets/coco_reid_unsup_prw.py',
    '../_base_/schedules/schedule_1x_reid_norm_base.py', '../_base_/default_runtime.py'
]
TEST = True
USE_PART_FEAT = True
GLOBAL_WEIGHT = 0.9
CO_LEARNING = False
UNCERTAINTY = True
HARD_MINING = True
USE_GFN = False
model = dict(
    roi_head=dict(
        use_gfn=USE_GFN,
        use_RoI_Align_feat=False,
        use_part_feat=USE_PART_FEAT,
        scene_emb_size=56,
        bbox_head=dict(
            testing=TEST,
            type='CGPSHead',
            id_num=18048,
            rcnn_bbox_bn=True,
            cluster_top_percent=0.6,
            instance_top_percent=1.0,
            use_quaduplet_loss=True,
            use_cluster_hard_loss=True,
            use_instance_hard_loss=False,
            use_IoU_loss=False,
            use_IoU_memory=False,
            use_uncertainty_loss=False,
            use_hard_mining=False,
            norm_type='protonorm',    # ['l2norm', 'protonorm', 'batchnorm']
            co_learning=CO_LEARNING,
            IoU_loss_clip=[0.7, 1.0],
            IoU_memory_clip=[0.05, 0.9],
            IoU_momentum=0.2,
            momentum=0.2,
            co_learning_weight=0.3,
            use_part_feat=USE_PART_FEAT,
            global_weight= GLOBAL_WEIGHT if USE_PART_FEAT else 1,
            use_hybrid_loss=False,
            use_instance_loss=False,
            use_inter_loss=False,
            triplet_instance_weight=1,
            num_features=256,
            margin=0.3,
            loss_bbox=dict(type='L1Loss', loss_weight=1),
            loss_reid=dict(loss_weight=1.0),
            gfn_config=dict(
                use_gfn=USE_GFN,
                gfn_mode='image',    # {'image', 'separate', 'combined'}
                gfn_activation_mode='se',   # combined:{'se', 'sum', 'identity'}
                gfn_filter_neg=True,
                gfn_query_mode='batch', # {'batch', 'oim'}
                gfn_use_image_lut=True,
                gfn_train_temp=0.1,
                gfn_se_temp=0.2,
                gfn_num_sample=(1, 1),
                emb_dim=2048,
            )
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
total_epochs = 36

SPCL=True
PSEUDO_LABELS = dict(
    freq=1, # epochs
    use_outliers=True,
    norm_feat=True,
    norm_center=True,
    SpCL=True,
    cluster='FINCH_context_SpCL_Plus',   # dbscan_context, FINCH_context, FINCH_context_SpCL, FINCH_context_SpCL_Plus
    eps=[0.68, 0.7, 0.72],
    min_samples=4, # for dbscan
    dist_metric='jaccard',
    k1=30, # for jaccard distance
    k2=6, # for jaccard distance
    search_type=0, # 0,1,2 for GPU, 3 for CPU (work for faiss)
    cluster_num=None,
    iters=1,    # 1
    lambda_scene=0,   # 调成0,即zero初始化
    lambda_person=0.1,
    context_method='scene',
    # context_clip=False,
    threshold=0.5,
    use_post_process=False,
    filter_threshold=0.2,
    use_crop=False,
    use_k_reciprocal_nearest=False,
    part_feat=dict(use_part_feat=USE_PART_FEAT, 
                    global_weight=GLOBAL_WEIGHT,
                    part_weight=(1 - GLOBAL_WEIGHT)/2,
                    uncertainty=UNCERTAINTY,
                    ),
    hard_mining=dict(use_hard_mining=HARD_MINING,
                    uncertainty_threshold=0.5,
                    label_refine_iters=0,
                    refine_global_weight=0.5
                    ),
    K=10,
)
# fp16 = dict(loss_scale=512.)
workflow = [('train', 1)]
evaluation = dict(start=16, interval=2, metric='bbox')
testing = TEST
save_features = True
restart = False