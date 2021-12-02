_base_ = '/mmdetection/configs/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco.py'
model = dict(
    roi_head = dict(
        bbox_head=dict(num_classes=6)))

dataset_type = 'MyDataset'
data_root = '/general_dataset/'
annot_root = '/general_dataset/groundtruth/json/sub_frames/'
tile_size= (500,500)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(type='VerticalFlip', p=0.5),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='RandomRotate90', p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.2),
    dict(type='Blur', blur_limit=15, p=0.2)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=tile_size, keep_ratio=True),
    dict(
        type='Albu',
        transforms=[
            dict(type='VerticalFlip', p=0.5),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='RandomRotate90', p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.2),
            dict(type='Blur', blur_limit=15, p=0.2)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]
val_pipeline  = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=tile_size,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=tile_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='MyDataset',
        ann_file=
        '/general_dataset/sub_frames_500/train/coco_subframes.json',
        img_prefix='/general_dataset/sub_frames_500/train',
        pipeline = train_pipeline),
    val=dict(
        type='MyDataset',
        ann_file=
        '/general_dataset/sub_frames_500/val/coco_subframes.json',
        img_prefix='/general_dataset/sub_frames_500/val',
        pipeline= val_pipeline),
    test=dict(
        type='MyDataset',
        ann_file=
        '/general_dataset/sub_frames_500/test/coco_subframes.json',
        img_prefix='/general_dataset/sub_frames_500/test',
        pipeline = test_pipeline))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(_delete_=True,max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.3333333333333333,
    step=[10, 20])
checkpoint_config = dict(create_symlink=False, interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
runner = dict(type='EpochBasedRunner', max_epochs=12)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/libra_faster_rcnn_r101_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
model_name= 'libra-cnn'
wnb_username= 'sfoucher'
#log_config = dict(interval=50,    hooks=[dict(type='TextLoggerHook'), dict(type='WandbLoggerHook')])
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='WandbLoggerHook',
                init_kwargs=dict(
                    project='fine-tuning-libra-cnn',
                    name=f'normalized {model_name}-workers{data["workers_per_gpu"]}samples/GPU{data["samples_per_gpu"]} lr {optimizer["lr"]}',
                    entity=wnb_username)
                )]
)
