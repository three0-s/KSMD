_base_ = '/home/jovyan/yewon/CV/PITR/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=7)
        ))

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

# We can use the pre-trained Faster RCNN model to obtain higher performance
load_from = '/home/jovyan/yewon/CV/PITR/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'