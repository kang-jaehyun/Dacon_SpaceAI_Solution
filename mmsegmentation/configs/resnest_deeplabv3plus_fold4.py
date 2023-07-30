norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[
        87.33,
        91.29,
        83.01,
    ],
    std=[
        43.75,
        38.6,
        35.43,
    ],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[
            87.33,
            91.29,
            83.01,
        ],
        std=[
            43.75,
            38.6,
            35.43,
        ],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        )),
    pretrained='open-mmlab://resnest101',
    backbone=dict(
        type='ResNeSt',
        depth=101,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        dilations=(
            1,
            1,
            2,
            4,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0),
            dict(
                type='LovaszLoss',
                loss_type='multi_class',
                classes=[
                    1,
                ],
                reduction='none',
                loss_weight=1.0),
        ]),
    auxiliary_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=0.4),
            dict(
                type='LovaszLoss',
                loss_type='multi_class',
                classes=[
                    1,
                ],
                reduction='none',
                loss_weight=0.4),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'SatelliteDataset'
data_root = '../datasets/Satellite'  # Root path of data.
crop_size = (
    256,
    256,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomCrop', crop_size=(
        256,
        256,
    )),
    dict(type='Resize', scale=(
        512,
        512,
    ), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(
        512,
        512,
    ), keep_ratio=True),
    dict(type='PackSegInputs'),
]
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=0.5, keep_ratio=True),
                dict(type='Resize', scale_factor=0.75, keep_ratio=True),
                dict(type='Resize', scale_factor=1.0, keep_ratio=True),
                dict(type='Resize', scale_factor=1.25, keep_ratio=True),
                dict(type='Resize', scale_factor=1.5, keep_ratio=True),
                dict(type='Resize', scale_factor=1.75, keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ]),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='SatelliteDataset',
        data_root = '../datasets/Satellite',  # Root path of data.
        data_prefix=dict(
            img_path='img_dir/train_4',
            seg_map_path='ann_dir/train_4'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='RandomCrop', crop_size=(
                256,
                256,
            )),
            dict(type='Resize', scale=(
                512,
                512,
            ), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs'),
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SatelliteDataset',
        data_root = '../datasets/Satellite',  # Root path of data.
        data_prefix=dict(
            img_path='img_dir/val_slice_4',
            seg_map_path='ann_dir/val_slice_4'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                512,
                512,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs'),
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SatelliteDataset',
        data_root = '../datasets/data/Satellite',  # Root path of data.
        data_prefix=dict(
            img_path='img_dir/test'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                512,
                512,
            ), keep_ratio=True),
            dict(type='PackSegInputs'),
        ]))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mIoU',
        'mDice',
    ])
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=[
        'mIoU',
    ],
    format_only=True,
    output_dir='mask_inference_result/resnest_deeplabv3plus_fold4/format_results')
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        # dict(type='TensorboardVisBackend'),
    ],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = True
tta_model = dict(type='SegTTAModel')
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(
        0.9,
        0.999,
    ), weight_decay=0.005)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(
            0.9,
            0.999,
        ), weight_decay=0.005),
    clip_grad=None,
    loss_scale='dynamic')
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-06,
        power=1.0,
        begin=0,
        end=500000,
        by_epoch=False),
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=150000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1000))
test_img_size = (
    512,
    512,
)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(
        512,
        512,
    ), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]
randomness = dict(seed=777)
launcher = 'none'
work_dir = '_satellite_new/resnest_test'
