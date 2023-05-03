# The new config inherits a base config to highlight the necessary modification
_base_ = [
    './mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
    './mmdetection/configs/_base_/datasets/coco_detection.py',
    './mmdetection/configs/_base_/schedules/schedule_2x.py', 
    './mmdetection/configs/_base_/default_runtime.py'
]


# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('fingerprint',)
data = dict(
    train=dict(
        type=dataset_type,
        img_prefix='',
        classes=classes,
        ann_file='./coco_fingerprint_TRAIN.json'),
    val=dict(
        type=dataset_type,
        img_prefix='',
        classes=classes,
        ann_file='./coco_fingerprint_TEST.json'),
    test=dict(
        type=dataset_type,
        img_prefix='',
        classes=classes,
        ann_file='./coco_fingerprint_TEST.json'))