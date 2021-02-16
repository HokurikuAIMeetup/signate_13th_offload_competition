norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.5,
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.5,
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
dataset_type = 'OffLoadDataset'
data_root = '/content/drive/MyDrive/signate/offload/data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (257, 257)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.2, 1.2)),
    dict(type='RandomCrop', crop_size=(257, 257), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(257, 257), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        img_ratios=[0.5, 0.75, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type='OffLoadDataset',
        data_root='/content/drive/MyDrive/signate/offload/data',
        img_dir=['train_images_A', 'leftImg8bit/train/bochum'],
        ann_dir=['train_annotations_A_4', 'tmp'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(1920, 1080), ratio_range=(0.2, 1.2)),
            dict(type='RandomCrop', crop_size=(257, 257), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(257, 257), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split=['splits/train.txt', 'splits/city.txt']),
    val=dict(
        type='OffLoadDataset',
        data_root='/content/drive/MyDrive/signate/offload/data',
        img_dir='train_images_A',
        ann_dir='train_annotations_A_4',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1920, 1080),
                img_ratios=[0.5, 0.75, 1.0],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/val.txt'),
    test=dict(
        type='OffLoadDataset',
        data_root='/content/drive/MyDrive/signate/offload/data',
        img_dir='precision_test_images',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1920, 1080),
                img_ratios=[0.5, 0.75, 1.0],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/test.txt'))
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/content/drive/MyDrive/signate/offload/weight/deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(
    interval=200,
    create_symlink=False,
    meta=dict(
        CLASSES=('bg', 'road', 'dirt road', 'other obstacle'),
        PALETTE=[[0, 0, 0], [128, 64, 128], [255, 128, 128], [0, 0, 70]]))
evaluation = dict(interval=200, metric='mIoU')
work_dir = '/content/drive/MyDrive/signate/offload/共有用/No27/weight'
total_iters = 200
seed = 2
gpu_ids = range(0, 1)
