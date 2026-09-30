# ============================================================================
# SSD-DEIM Configuration for GC2026 Dataset
# Paper: SSD-DEIM: Preserving Multi-Scale Fidelity and Geometric Priors
#         for Dense Camouflaged Small Object Detection (ICASSP 2027)
# ============================================================================

# ---------------------------------------------------------------------------
#  Model Architecture
# ---------------------------------------------------------------------------

model = dict(
    type='SSDDEIM',

    # ---- Backbone: HGNetV2 + DFD (Duplex Feature Downsampling) ----
    backbone=dict(
        type='HGNetV2DFD',
        name='n',               # Model size: 'n', 's', 'm', 'l', 'x'
        return_idx=[1, 2, 3],   # Output stages: S3(stride8), S4(stride16), S5(stride32)
        act='relu',             # Activation function
        agg='ese',              # Feature aggregation: 'ese', 'se', 'none'
        pretrained=False,       # Load pretrained weights
        use_dfd=True,           # Enable DFD module
        drop_path_rate=0.0,     # Drop path rate
    ),

    # ---- Neck: BFF (Bidirectional Feature Fusion) ----
    neck=dict(
        type='BFFNeck',
        # in_channels will be auto-filled from backbone output channels
        # For HGNetV2-n: [256, 512, 1024] for stages 1,2,3
        in_channels=[256, 512, 1024],
        out_channels=[128, 256, 512],   # Output channels for P3, P4, P5
        num_c2f_blocks=2,               # Number of PCGU blocks per C2f
        partial_ratio=0.25,             # Active channel ratio in PCGU (1/4)
        use_bfe=True,                   # Enable BFE (Bidirectional Feature Enhancement)
    ),

    # ---- Head: DFINE-style Detection Head ----
    head=dict(
        type='DFINEHead',
        head_module=dict(
            type='DFINEHead',
            # in_channels will be auto-filled from neck output channels
            in_channels=[128, 256, 512],
            feat_channels=256,        # Decoder embedding dimension
            num_classes=1,            # Single class: citrus fruit
            num_decoder_layers=6,     # Number of decoder layers
            num_heads=8,              # Attention heads
            num_points=4,             # Deformable attention points per level
            reg_max=16,               # Distribution bins for integral regression
            reg_scale=2.0,            # Weighting function scale
            dim_feedforward=1024,     # FFN dimension
            dropout=0.0,              # Dropout rate
            activation='relu',        # Activation in decoder
        ),

        # ---- Classification Loss ----
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,               # Focal loss alpha
            gamma=2.0,                # Focal loss gamma
            iou_weighted=True,        # IoU-aware quality weighting
            loss_weight=1.0,          # Loss weight
        ),

        # ---- Bbox Regression Loss: GPL (Focaler-ShapeIoU) ----
        loss_bbox=dict(
            type='GPLLoss',
            iou_mode='shapeiou',      # IoU type: 'shapeiou', 'ciou', 'siou', etc.
            scale=0.0,                # ShapeIoU scale parameter
            eps=1e-7,                 # Numerical stability
            loss_weight=2.0,          # Overall loss weight
            # Focaler parameters (hard sample focusing)
            d=0.00,                   # Lower IoU bound for critical range
            u=0.95,                   # Upper IoU bound for critical range
            gamma=2.0,                # Focaler focusing strength
        ),

        # ---- Training Config ----
        train_cfg=dict(
            assigner=dict(
                type='DenseODMatcher',
                topk=13,               # Top-k candidates per GT
                cost_class=2.0,        # Classification cost weight
                cost_bbox=5.0,         # L1 box distance cost weight
                cost_giou=2.0,         # GIoU cost weight
                use_focal_loss=True,   # Use focal loss for class cost
                alpha=0.25,            # Focal alpha
                gamma=2.0,             # Focal gamma
                use_iou_order=True,    # Use IoU-ordered matching
                iou_order_alpha=1.0,   # IoU order alpha
            ),
            aux_loss_weight=0.8,      # Auxiliary loss decay factor
        ),

        # ---- Test Config ----
        test_cfg=dict(
            max_per_img=300,
            score_thr=0.01,
            nms_thr=0.5,
        ),
    ),
)


# ---------------------------------------------------------------------------
#  Dataset Settings
# ---------------------------------------------------------------------------

dataset_type = 'CocoDataset'
data_root = 'datasets/GC2026/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

img_scale = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies=[
            [dict(type='RandomCrop', crop_size=(512, 512))],
            [dict(type='PhotoMetricDistortion')],
            [dict(
                type='RandomAffine',
                max_rotate_degree=10.0,
                max_translate_ratio=0.1,
                max_scaling_ratio=0.5,
                max_shear_degree=2.0,
            )],
        ],
    ),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=16,          # Batch size per GPU
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
    ),
)


# ---------------------------------------------------------------------------
#  Optimizer & Scheduler
# ---------------------------------------------------------------------------

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    weight_decay=1e-4,
    betas=(0.937, 0.999),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,          # ~5 epochs at ~300 iters/epoch
    warmup_ratio=0.001,
    min_lr_ratio=1e-6,
)

runner = dict(type='EpochBasedRunner', max_epochs=200)


# ---------------------------------------------------------------------------
#  Training Hooks
# ---------------------------------------------------------------------------

checkpoint_config = dict(interval=10, max_keep_ckpts=3)
evaluation = dict(interval=5, metric='bbox', save_best='bbox_mAP')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook'),
    ],
)

# Mixed precision training
fp16 = dict(
    enabled=False,               # Set to True to enable AMP
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)


# ---------------------------------------------------------------------------
#  Other Settings
# ---------------------------------------------------------------------------

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'work_dirs/ssd_deim_gc2026/'
seed = 42
gpu_ids = range(1)
deterministic = False

# EMA (Exponential Moving Average)
ema = dict(
    enabled=False,
    decay=0.9999,
    warmup_steps=1000,
)
