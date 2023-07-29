########################################
########## model setting ###############
########################################

# model settings
seed = 777
norm_cfg = dict(type='SyncBN', requires_grad=True)  # Segmentation usually uses SyncBN

crop_size = (224, 224)
valid_crop_size = (256, 256)
resize_ratio = 1.5
stride_size = (int((valid_crop_size[0] - crop_size[0]) * resize_ratio), int((valid_crop_size[0] - crop_size[0]) * resize_ratio))

data_preprocessor = dict(  # The config of data preprocessor, usually includes image 
    type='SegDataPreProcessor',  # The type of data preprocessor.
    mean=[87.33, 91.29, 83.01],
    std=[43.75, 38.60, 35.43],
    bgr_to_rgb=True,  # Whether to convert image from BGR to RGB.
    pad_val=0,  # Padding value of image.
    seg_pad_val=255,  # Padding value of segmentation map.
    size=(int(crop_size[0] * resize_ratio), int(crop_size[1] * resize_ratio)))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='BEiT',
        img_size=(int(crop_size[0] * resize_ratio), int(crop_size[1] * resize_ratio)),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=0.1),
        
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),

    decode_head=dict(
        type='UPerHead',  # Type of decode head
        in_channels=[768, 768, 768, 768],  # Input channel of decode head.
        in_index=[0, 1, 2, 3],  # The index of feature map to select.
        pool_scales=(1, 2, 3, 6),
        channels=768,  # The intermediate channels of decode head.
        dropout_ratio=0.1,  # The dropout ratio before final classification layer.
        num_classes=2,  # Number of segmentation class.
        norm_cfg=norm_cfg,
        align_corners=False,  # The align_corners argument for resize in decoding.
        loss_decode=[  # Type of loss used for segmentation.
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='LovaszLoss', loss_type='multi_class', classes=[1], reduction='none', loss_weight=1.0 * 1.0),
        ],
    ),

    auxiliary_head=[
        dict(
            type='DepthwiseSeparableASPPHead',  # Type of decode head
            in_channels=768,  # Input channel of decode head.
            in_index=2,  # The index of feature map to select.els of decode head.
            dilations=(1, 12, 24, 36),
            channels=512,  # The intermediate chann
            c1_in_channels=768,
            c1_channels=192,
            dropout_ratio=0.1,  # The dropout ratio before final classification layer.
            num_classes=2,  # Number of segmentation class.
            norm_cfg=norm_cfg,
            align_corners=False,  # The align_corners argument for resize in decoding.
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.4),
                dict(type='LovaszLoss', loss_type='multi_class', classes=[1], reduction='none', loss_weight=1.0 * 0.4),
            ],
        ),
        dict(
            type='DepthwiseSeparableASPPHead',  # Type of decode head
            in_channels=768,  # Input channel of decode head.
            in_index=1,  # The index of feature map to select.
            channels=512,  # The intermediate channels of decode head.
            dilations=(1, 6, 12, 18),
            c1_in_channels=768,
            c1_channels=192,
            dropout_ratio=0.1,  # The dropout ratio before final classification layer.
            num_classes=2,  # Number of segmentation class.
            norm_cfg=norm_cfg,
            align_corners=False,  # The align_corners argument for resize in decoding.
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.4),
                dict(type='LovaszLoss', loss_type='multi_class', classes=[1], reduction='none', loss_weight=1.0 * 0.4),
            ],
        ),
        # dict(
        #     type='DepthwiseSeparableASPPHead',  # Type of decode head
        #     in_channels=768,  # Input channel of decode head.
        #     in_index=0,  # The index of feature map to select.
        #     channels=512,  # The intermediate channels of decode head.
        #     dilations=(1, 3, 6, 9),
        #     c1_in_channels=768,
        #     c1_channels=192,
        #     dropout_ratio=0.1,  # The dropout ratio before final classification layer.
        #     num_classes=2,  # Number of segmentation class.
        #     norm_cfg=norm_cfg,
        #     align_corners=False,  # The align_corners argument for resize in decoding.
        #     loss_decode=[
        #         dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.4),
        #         dict(type='LovaszLoss', loss_type='multi_class', classes=[1], reduction='none', loss_weight=1.0 * 0.4),
        #     ],
        # ),
    ],
    # model training and testing settings
    train_cfg=dict(), 
    test_cfg=dict(
        mode='slide',
        crop_size=(int(crop_size[0] * resize_ratio), int(crop_size[1] * resize_ratio)),
        stride=stride_size
    ),
)


########################################
########## dataset setting #############
########################################


# dataset settings
dataset_type = 'SatelliteDataset'  # Dataset type, this will be used to define the dataset.
data_root = 'data/Satellite'  # Root path of data.
train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Second pipeline to load annotations for current image.

    # Augmentation pipeline that resize the images and their annotations.
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.90),
    dict(type='Resize', scale=(int(crop_size[0] * resize_ratio), int(crop_size[1] * resize_ratio)), keep_ratio=True),
    # dict(
    #     type='Albu',
    #     transforms=[
    #         dict(type='OneOf', transforms=[
    #             dict(type='ColorJitter', brightness=0.1, contrast=0.15, saturation=0.2, hue=0.2, p=0.5),
    #             dict(type='ElasticTransform', alpha=1, sigma=50, alpha_affine=50, p=0.3),
    #             dict(type='RandomGamma', gamma_limit=(80, 100), p=0.4),
    #         ], p=0.3),
    #     ],
    # ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize', scale=(int(valid_crop_size[0] * resize_ratio), int(valid_crop_size[1] * resize_ratio)), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize', scale=(int(crop_size[0] * resize_ratio), int(crop_size[1] * resize_ratio)), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackSegInputs')
]

img_ratios = [1.5, 1.75, 2.0, 3.0]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='PackSegInputs')]
        ])
]


train_dataloader = dict(
    batch_size=16,  # Batch size of a single GPU
    num_workers=4,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train_4', seg_map_path='ann_dir/train_4'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing.
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val_slice_4', seg_map_path='ann_dir/val_slice_4'),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing.
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/test'),
        pipeline=test_pipeline))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])

# TODO: test evaluator for competition
test_evaluator = dict(
    type='CityscapesMetric',
    format_only=True,
    keep_results=True,
    output_dir='_satellite/beit-b_upernet_ver8_fold4/format_results')



########################################
############## scheduler ###############
########################################

# optimizer
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        power=1.0,
        begin=0,
        end=100000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, max_keep_ckpts=1, save_best='mDice'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=250))



########################################
########## default runtime #############
########################################

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
                # dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'

# load pretrained model from mmseg
load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-base_8x2_640x640_160k_ade20k/upernet_beit-base_8x2_640x640_160k_ade20k-eead221d.pth"

tta_model = dict(type='SegTTAModel')