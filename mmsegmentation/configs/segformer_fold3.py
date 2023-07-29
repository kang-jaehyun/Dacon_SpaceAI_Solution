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
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        )),
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[
            3,
            8,
            27,
            3,
        ],
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'
        )),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        channels=256,
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
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'SatelliteDataset'
data_root = 'data/Satellite'
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
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='SatelliteDataset',
        data_root='data/Satellite',
        data_prefix=dict(
            img_path='img_dir/fold3_train_img',
            seg_map_path='ann_dir/fold3_train_gt'),
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
        data_root='data/Satellite',
        data_prefix=dict(
            img_path='img_dir/fold3_val_img',
            seg_map_path='ann_dir/fold3_val_gt'),
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
        data_root='data/Satellite',
        data_prefix=dict(
            img_path='img_dir/test_img',
            seg_map_path='annotations/validation'),
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
    output_dir='_result_img/segformer_fold0_numclass2_lossadded')
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = True
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(
            0.9,
            0.999,
        ), weight_decay=0.001),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))),
    loss_scale='dynamic')
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=5e-06,
        power=1.0,
        begin=0,
        end=100000,
        by_epoch=False),
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1000))
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'
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
work_dir = '_final/segforemr_f3'
