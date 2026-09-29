# SSD-DEIM Configuration for GC2026 Dataset
# Paper: SSD-DEIM: Preserving Multi-Scale Fidelity and Geometric Priors
#         for Dense Camouflaged Small Object Detection (ICASSP 2027)

model = dict(
    type='SSDDEIM',
    backbone=dict(
        type='CSPDarknetDFD',
        depth=0.33,
        width=0.25,
        out_indices=(1, 2, 3),
        dfd_channels=[64, 128, 256],  # DFD module channels for each stage
        frozen_stages=-1,
        use_depthwise=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
        )
    ),
    neck=dict(
        type='BFFPAN',
        in_channels=[128, 256, 512],
        out_channels=[128, 256, 512],
        num_extra_trans=1,
        use_partial_conv=True,  # PCGU: Partial Convolution Gated Unit
        partial_ratio=0.25,    # Active channel ratio (C/4)
    ),
    head=dict(
        type='SSDDEIMHead',
        head_module=dict(
            type='SSDDEIMHeadModule',
            feat_channels=128,
            num_classes=1,  # single class: citrus fruit
        ),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='GPLoss',  # Morphology-Aware Geometric Loss
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=2.0,
            # GPL specific parameters
            d=0.00,           # IoU lower bound for hard sample reweighting
            u=0.95,           # IoU upper bound
            shape_weight=0.5, # Omega_shape weight
            use_aspect_ratio_reg=True,
        ),
        train_cfg=dict(
            assigner=dict(
                type='DualODAssigner',
                topk=13,
                max_epochs=200,  # total epochs
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        test_cfg=dict(
            max_per_img=300,
            score_thr=0.01,
            nms=None,  # NMS-free
        )
    )
)

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'datasets/GC2026/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies=[
            [dict(type='RandomCrop', crop_size=(512, 512))],
            [dict(type='PhotoMetricDistortion')],
            [dict(type='RandomAffine',
                  max_rotate_degree=10.0,
                  max_translate_ratio=0.1,
                  max_scaling_ratio=0.5,
                  max_shear_degree=2.0)]
        ]
    ),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(640, 640),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img']),
         ])
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=4,  # batch size per GPU (total 16 for 4 GPUs)
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline,
    )
)

# Training settings
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    weight_decay=1e-4,
    betas=(0.937, 0.999),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2)
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,  # 5 epochs * ~300 iters
    warmup_ratio=0.001,
    min_lr_ratio=1e-6,
)

runner = dict(type='EpochBasedRunner', max_epochs=200)

checkpoint_config = dict(interval=10, max_keep_ckpts=3)
evaluation = dict(interval=5, metric='bbox', save_best='bbox_mAP')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook'),
    ]
)

# Other settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'work_dirs/ssd_deim_gc2026/'
seed = 42
gpu_ids = range(4)
