_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmdet.core.bbox.assigners.point_assigner_v2'],
    allow_failed_imports=False)
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth' 
model = dict(
    type='RepPointsV2Detector',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
     bbox_head=dict(
        type='RepPointsV2Head',
        num_classes=7,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        shared_stacked_convs=1,
        first_kernel_size=3,
        kernel_size=1,
        corner_dim=64,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=0.25),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_sem=dict(
            type='SEPFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.1),
        transform_method='exact_minmax'),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssignerV2', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
         heatmap=dict(
            assigner=dict(type='PointHMAssigner', gaussian_bump=True, gaussian_iou=0.7),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)), 
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('rain', 'cloud', 'person', 'puddle', 'lightning', 'direct protectection', 'indirect protection')
data = dict(
    train=dict(
        img_prefix='/home/jovyan/yewon/CV/PITR/img_train/',
        classes=classes,
        ann_file='/home/jovyan/yewon/CV/PITR/pitr_train_annotation.json'),
    val=dict(
        img_prefix='/home/jovyan/yewon/CV/PITR/img_valid/',
        classes=classes,
        ann_file='/home/jovyan/yewon/CV/PITR/pitr_valid_annotation.json'),
    test=dict(
        img_prefix='/home/jovyan/yewon/CV/PITR/img_valid/',
        classes=classes,
        ann_file='/home/jovyan/yewon/CV/PITR/pitr_valid_annotation.json'),
   )
load_from = '/home/jovyan/yewon/CV/PITR/mmdetection/checkpoints/reppoints_v2_x101_fpn_dconv_c3-c5_giou_mstrain_2x_coco-3d418239.pth'