# %%writefile /content/Masterclass-Biodiversity/jupyter_notebooks/config2.py

# Setting the base model to use.
_base_ = "/mmdetection/configs/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco.py"
# _base_ = '/content/mmdetection/configs/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco.py'

# Setting the number of classes to be analysed by the chosen model.
model = dict(
    roi_head = dict(
        bbox_head = dict(
            num_classes = 6  #TODO: Remove hard coded values.
            # # TODO: Move In section 3) of this script.
            # loss_cls=dict(
            #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[? ? ? ? ? ?]),
        )
    )
)

# Setting the name and paths of the dataset.
dataset_type = "MyDataset"
data_root = "/general_dataset/sub_frames_500/"
# data_root = "/content/sub_frames/"
annot_root = data_root

# Setting the size of the previously created subframes.
tile_size = (500, 500)  #TODO: Remove hard coded values.
# tile_size= (1000,1000)

# Setting the image configuration TODO: Verify this section of the script.
img_norm_cfg = dict(
    mean = [123.675, 116.28, 103.53],
    std = [58.395, 57.12, 57.375],
    to_rgb = True
)

# Setting data augmentation parameters with the "albumentations" Python module for the training dataset. TODO: Verify this section of the script.
albu_train_transforms = [
    dict(
        type = "VerticalFlip",
        p = 0.5
    ),
    dict(
        type = "HorizontalFlip",
        p = 0.5
    ),
    dict(
        type = "RandomRotate90",
        p = 0.5
    ),
    dict(
        type = "RandomBrightnessContrast",
        brightness_limit = 0.2,
        contrast_limit = 0.2,
        p = 0.2
    ),
    dict(
        type = "Blur",
        blur_limit = 15,
        p = 0.2
    )
]

# Setting the pipeline for the training dataset.
train_pipeline = [
    dict(
        type = "LoadImageFromFile"
    ),
    dict(
        type = "LoadAnnotations",
        with_bbox = True
    ),
    dict(
        type = "Resize",
        img_scale = tile_size,
        keep_ratio = True
    ),
    # The following information are the "albu_train_transforms" variable.
    dict(
        type = "Albu",
        transforms = [
            dict(
                type = "VerticalFlip",
                p = 0.5
            ),
            dict(
                type = "HorizontalFlip",
                p = 0.5
            ),
            dict(
                type = "RandomRotate90",
                p = 0.5
            ),
            dict(
                type = "RandomBrightnessContrast",
                brightness_limit = 0.2,
                contrast_limit = 0.2,
                p = 0.2
            ),
            dict(
                type = "Blur",
                blur_limit = 15,
                p = 0.2
            )
        ],
        bbox_params = dict(
            type = "BboxParams",
            format = "pascal_voc",
            label_fields = [
                "gt_labels"
            ],
            min_visibility = 0.0
        ),
        keymap = dict(
            img = "image",
            gt_bboxes = "bboxes"
        ),
        update_pad_shape = False,
        skip_img_without_anno = True
    ),
    # The following information are the "img_norm_cfg" variable.
    dict(
        type = "Normalize",
        mean = [123.675, 116.28, 103.53],
        std = [58.395, 57.12, 57.375],
        to_rgb = True
    ),
    dict(
        type = "Pad",
        size_divisor = 32
    ),
    dict(
        type = "DefaultFormatBundle"
    ),
    dict(
        type = "Collect",
        keys = [
            "img",
            "gt_bboxes",
            "gt_labels"
        ],
        meta_keys = (
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor"
        )
    )
]

# Setting the pipeline for the validation dataset.
val_pipeline = [
    dict(
        type = "LoadImageFromFile"
    ),
    dict(
        type = "MultiScaleFlipAug",
        img_scale = tile_size,
        flip = False,
        transforms = [
            dict(
                type = "Resize",
                keep_ratio = True
            ),
            dict(
                type = "RandomFlip"
            ),
            dict(
                type = "Normalize",
                mean = [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_rgb = True
            ),
            dict(
                type = "Pad",
                size_divisor = 32
            ),
            dict(
                type = "ImageToTensor",
                keys = [
                    "img"
                ]
            ),
            dict(
                type = "Collect",
                keys = [
                    "img"
                ]
            )
        ]
    )
]

# Setting the pipeline for the testing dataset.
test_pipeline = [
    dict(
        type = "LoadImageFromFile"
    ),
    dict(
        type = "MultiScaleFlipAug",
        img_scale = tile_size,
        flip = False,
        transforms = [
            dict(
                type = "Resize",
                keep_ratio = True
            ),
            dict(
                type = "RandomFlip"
            ),
            dict(
                type = "Normalize",
                mean = [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_rgb = True
            ),
            dict(
                type = "Pad",
                size_divisor = 32
            ),
            dict(
                type = "ImageToTensor",
                keys = [
                    "img"
                ]
            ),
            dict(
                type = "Collect",
                keys = [
                    "img"
                ]
            )
        ]
    )
]

# TODO: Verify this section of the script.
data = dict(
    samples_per_gpu = 2,  #Batch size. par GPU.
    workers_per_gpu = 2,  #TODO: to be verified. par CPU.
    train = dict(
        type = "MyDataset",
        # ann_file = annot_root + "train/coco_subframes.json",
        ann_file = annot_root + "train_v2_TEST/coco_subframes.json",
        # img_prefix = data_root + "train",
        img_prefix = data_root + "train_v2_TEST",
        pipeline = train_pipeline
    ),
    val = dict(
        type = "MyDataset",
        # ann_file = annot_root + "val/coco_subframes.json",
        ann_file = annot_root + "val_TEST/coco_subframes.json",
        # img_prefix = data_root + "val",
        img_prefix = data_root + "val_TEST",
        pipeline = val_pipeline
    ),
    test = dict(
        type = "MyDataset",
        # ann_file = annot_root + "test/coco_subframes.json",
        ann_file = annot_root + "test_TEST/coco_subframes.json",
        # img_prefix = data_root + "test",
        img_prefix = data_root + "test_TEST",
        pipeline = test_pipeline
    )
)

# TODO: Verify this section of the script.
evaluation = dict(
    interval = 1,
    metric = "bbox"
)

# TODO: Verify this section of the script.
optimizer = dict(
    type = "SGD",
    lr = 0.001,
    momentum = 0.9,
    weight_decay = 0.001
)

# TODO: Verify this section of the script.
optimizer_config = dict(
    grad_clip = dict(
        _delete_ = True,
        max_norm = 35,
        norm_type = 2
    )
)

# TODO: Verify this section of the script.
lr_config = dict(
    policy = "step",
    warmup = "linear",
    warmup_iters = 100,
    warmup_ratio = 0.3333333333333333,
    step = [10, 20]
)

# TODO: Verify this section of the script.
checkpoint_config = dict(
    create_symlink = False,
    interval = 1
)

# TODO: Verify this section of the script.
log_config = dict(
    interval = 50,
    hooks = [
        dict(
            type = "TextLoggerHook"
        )
    ]
)

# TODO: Verify this section of the script.
runner = dict(
    type = "EpochBasedRunner",
    max_epochs = 12
)

# TODO: Verify this section of the script.
dist_params = dict(
    backend = "nccl"
)

# TODO: Verify this section of the script.
log_level = "INFO"

# TODO: Verify this section of the script.
work_dir = "./work_dirs/libra_faster_rcnn_r101_fpn_1x"

# TODO: Verify this section of the script.
load_from = None

# TODO: Verify this section of the script.
resume_from = None

# TODO: Verify this section of the script.
workflow = [
    ("train", 1)
]

# TODO: Verify this section of the script.
model_name = "libra-cnn"

# TODO: Verify this section of the script.
wnb_username = "sdurand"
# wnb_username = "sfoucher"

# TODO: Verify this section of the script.
#log_config = dict(interval=50,    hooks=[dict(type='TextLoggerHook'), dict(type='WandbLoggerHook')])
log_config = dict(
    interval = 50,
    hooks = [
        dict(
            type = "TextLoggerHook"
        ),
        dict(
            type = "WandbLoggerHook",
            init_kwargs = dict(
                project = "fine-tuning-libra-cnn",
                name = f'normalized {model_name}-workers{data["workers_per_gpu"]}samples/GPU{data["samples_per_gpu"]} lr {optimizer["lr"]}',
                config = {
                    "learning_rate": optimizer["lr"],
                    "architecture": "libra-cnn"
                },
                entity = wnb_username
            )
        )
    ]
)
