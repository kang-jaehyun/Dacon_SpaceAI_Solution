crop_size = (224, 224)
valid_crop_size = (256, 256)
resize_ratio = 2
seed = 777
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[87.33, 91.29, 83.01],
    std=[43.75, 38.60, 35.43],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(int(crop_size[0] * resize_ratio), int(crop_size[1] * resize_ratio)))

norm_cfg = dict(type='SyncBN', requires_grad=True)
custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'  # noqa

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),

    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='LovaszLoss', loss_type='multi_class', classes=[1], reduction='none', loss_weight=1.0 * 1.0),
        ]
    ),

    auxiliary_head=[
        dict(
            type='DepthwiseSeparableASPPHead',  # Type of decode head
            in_channels=512,  # Input channel of decode head.
            in_index=2,  # The index of feature map to select.
            channels=512,  # The intermediate channels of decode head.
            dilations=(1, 12, 24, 36),
            c1_in_channels=128,
            c1_channels=48,
            dropout_ratio=0.1,  # The dropout ratio before final classification layer.
            num_classes=2,  # Number of segmentation class.
            norm_cfg=norm_cfg,
            align_corners=False,  # The align_corners argument for resize in decoding.
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.4),
                dict(type='LovaszLoss', loss_type='multi_class', classes=[1], reduction='none', loss_weight=1.0 * 0.4),
            ]
        ),
        dict(
            type='DepthwiseSeparableASPPHead',  # Type of decode head
            in_channels=256,  # Input channel of decode head.
            in_index=1,  # The index of feature map to select.
            channels=512,  # The intermediate channels of decode head.
            dilations=(1, 6, 12, 18),
            c1_in_channels=128,
            c1_channels=48,
            dropout_ratio=0.1,  # The dropout ratio before final classification layer.
            num_classes=2,  # Number of segmentation class.
            norm_cfg=norm_cfg,
            align_corners=False,  # The align_corners argument for resize in decoding.
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.4),
                dict(type='LovaszLoss', loss_type='multi_class', classes=[1], reduction='none', loss_weight=1.0 * 0.4),
            ]
        ),
        # dict(
        #     type='DepthwiseSeparableASPPHead',  # Type of decode head
        #     in_channels=128,  # Input channel of decode head.
        #     in_index=0,  # The index of feature map to select.
        #     channels=128,  # The intermediate channels of decode head.
        #     dilations=(1, 3, 6, 9),
        #     c1_in_channels=128,
        #     c1_channels=48,
        #     dropout_ratio=0.1,  # The dropout ratio before final classification layer.
        #     num_classes=2,  # Number of segmentation class.
        #     norm_cfg=norm_cfg,
        #     align_corners=False,  # The align_corners argument for resize in decoding.
        #     loss_decode=[
        #         dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.4),
        #         dict(type='LovaszLoss', loss_type='multi_class', classes=[1], reduction='none', loss_weight=1.0 * 0.4),
        #     ]
        # ),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



########################################
########## dataset setting #############
########################################


# dataset settings
dataset_type = 'SatelliteDataset'  # Dataset type, this will be used to define the dataset.
data_root = '../datasets/Satellite'  # Root path of data.
train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Second pipeline to load annotations for current image.

    # Augmentation pipeline that resize the images and their annotations.    
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.90),
    dict(type='Resize', scale=(crop_size[0] * resize_ratio, crop_size[1] * resize_ratio), keep_ratio=True),
    # dict(
    #     type='Albu',
    #     transforms=[
    #         dict(type='OneOf', transforms=[
    #             dict(type='ColorJitter', brightness=0.1, contrast=0.15, saturation=0.2, hue=0.2, p=0.3),
    #             dict(type='ElasticTransform', alpha=1, sigma=50, alpha_affine=50, p=0.3),
    #             dict(type='RandomGamma', gamma_limit=(80, 100), p=0.3),
    #         ], p=0.3),
    #     ],
    # ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize', scale=(valid_crop_size[0] * resize_ratio, valid_crop_size[1] * resize_ratio), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize', scale=(crop_size[0] * resize_ratio, crop_size[1] * resize_ratio), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='RandomFlip', prob=0., direction='vertical'),
                dict(type='RandomFlip', prob=1., direction='vertical')
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='PackSegInputs')]
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
    output_dir='mask_inference_result/convnext-base_fold4/format_results')



########################################
############## scheduler ###############
########################################

# optimizer
optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic',
)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        power=1.0,
        begin=0,
        end=150000,
        eta_min=0.0,
        by_epoch=False,
    )
]


# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=150000, val_interval=4000)
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
load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_640x640_160k_ade20k/upernet_convnext_base_fp16_640x640_160k_ade20k_20220227_182859-9280e39b.pth"

resume = False

tta_model = dict(type='SegTTAModel')