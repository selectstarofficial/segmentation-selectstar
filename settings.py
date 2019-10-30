"""CUDA_VISIBLE_DEVICES=1,2,3 python3 train.py"""

backbone = 'xception'
out_stride = 16  # network output stride (default: 8)
use_sbd = False  # whether to use SBD dataset
workers = 16

resize_height = 512
resize_width = 1024

cuda = True

# If you want to use gpu:1,2,3, run CUDA_VISIBLE_DEVICES=1,2,3 python3 ...
# with gpu_ids option [0,1,2] starting with zero
gpu_ids = [0, 1, 2, 3]  # use which gpu to train

sync_bn = True if len(gpu_ids) > 1 else False  # whether to use sync bn
freeze_bn = False  # whether to freeze bn parameters (default: False)

epochs = 200
start_epoch = 0
batch_size = 2 * len(gpu_ids)
test_batch_size = 2 * len(gpu_ids)

loss_type = 'ce'  # 'ce': CrossEntropy, 'focal': Focal Loss
use_balanced_weights = False  # whether to use balanced weights (default: False)
lr = 0.01
lr_scheduler = 'poly'  # lr scheduler mode: ['poly', 'step', 'cos']
momentum = 0.9
weight_decay = 5e-4
nesterov = True

resume = None  # put the path to resuming file if needed
checkname = "deeplab"  # set the checkpoint name

ft = False  # finetuning on a different dataset
eval_interval = 1  # evaluuation interval (default: 1)
no_val = False  # skip validation during training

dataset = 'surface'
root_dir = ''
if dataset == 'pascal':
    root_dir = '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
elif dataset == 'sbd':
    root_dir = '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
elif dataset == 'cityscapes':
    root_dir = '/path/to/datasets/cityscapes/'  # foler that contains leftImg8bit/
elif dataset == 'coco':
    root_dir = '/home/super/Projects/dataset/coco/'
elif dataset == 'surface':
    root_dir = '/home/super/Projects/dataset/surface'
else:
    print('Dataset {} not available.'.format(dataset))
    raise NotImplementedError


classes = [
"background",
"sidewalk@block",
"sidewalk@cement",
"sidewalk@urethane",
"sidewalk@asphalt",
"sidewalk@soil_stone",
"sidewalk@damaged",
"sidewalk@other",
"braille_guilde_block@normal",
"braille_guilde_block@damaged",
"roadway@normal",
"roadway@crosswalk",
"alley@normal",
"alley@crosswalk",
"alley@speed_bump",
"alley@damaged",
"bike_lane@normal",
"caution_zone@stairs",
"caution_zone@manhole",
"caution_zone@tree_zone",
"caution_zone@grating",
"caution_zone@repair_zone",
]

# RGB
colors = [
    [0, 0, 0],
    [0, 0, 255],
    [217, 217, 217],
    [198, 89, 17],
    [128, 128, 128],
    [255, 230, 153],
    [55, 86, 35],
    [110, 168, 70],
    [255, 255, 0],
    [128, 96, 0],
    [255, 128, 255],
    [255, 0, 255],
    [230, 170, 255],
    [208, 88, 255],
    [138, 60, 200],
    [88, 38, 128],
    [255, 155, 155],
    [255, 192, 0],
    [255, 0, 0],
    [0, 255, 0],
    [255, 128, 0],
    [105, 105, 255],
]

assert len(classes) == len(colors), "Color Class mapping mismatched."
num_classes = len(classes)