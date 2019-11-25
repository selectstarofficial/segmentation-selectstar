"""CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py"""

backbone = 'xception'
out_stride = 8  # network output stride (default: 8)
workers = 16
pretrained = True  # whether to use pretrained Xception backbone

resize_height = 512  # model input shape
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
lr = 1e-3

# Adam optimizer performed far better.
# lr_scheduler = 'poly'  # lr scheduler mode: ['poly', 'step', 'cos']
# momentum = 0.9
# weight_decay = 5e-4
# nesterov = False

resume = False  # True: load checkpoint model. False: train from scratch
checkpoint = './run/surface/deeplab/model_iou_77.pth.tar'

checkname = "deeplab"  # set the checkpoint name

ft = False  # finetuning on a different dataset
eval_interval = 1  # evaluuation interval (default: 1)
no_val = False  # skip validation during training

dataset = 'surface'
root_dir = ''
if dataset == 'pascal':
    use_sbd = False  # whether to use SBD dataset
    root_dir = '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
elif dataset == 'sbd':
    root_dir = '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
elif dataset == 'cityscapes':
    root_dir = '/path/to/datasets/cityscapes/'  # foler that contains leftImg8bit/
elif dataset == 'coco':
    root_dir = '/home/super/Projects/dataset/coco/'
elif dataset == 'surface':
    root_dir = '/home/super/Projects/dataset/surface6'
else:
    print('Dataset {} not available.'.format(dataset))
    raise NotImplementedError

"""
background	    0   [0, 0, 0]
bike_lane	    1   [255, 128, 0]
caution_zone	2   [255, 0, 0]
crosswalk	    3   [255, 0, 255]
guide_block	    4   [255, 255, 0]
roadway	        5   [0, 0, 255]
sidewalk	    6   [0, 255, 0]
"""
"""
Class	                Attr	    Unique	        Label	R	G	B
background		                    background	    0	    0	0	0
sidewalk	            blocks	    sidewalk	    6	    0	0	255
sidewalk	            cement	    sidewalk	    6	    217	217	217
sidewalk	            urethane	bike_lane	    1	    198	89	17
sidewalk	            asphalt	    background	    1	    128	128	128
sidewalk	            soil_stone	sidewalk	    6	    255	230	153
sidewalk	            damaged	    sidewalk	    6	    55	86	35
sidewalk	            other	    sidewalk	    6	    110	168	70
braille_guide_blocks	normal	    guide_block	    4	    255	255	0
braille_guide_blocks	damaged	    guide_block	    4	    128	96	0
roadway	                normal	    roadway	        5	    255	128	255
roadway	                crosswalk	crosswalk	    3	    255	0	255
alley	                normal	    roadway	        5	    230	170	255
alley	                crosswalk	crosswalk	    3	    208	88	255
alley	                speed_bump	roadway	        5	    138	60	200
alley	                damaged	    roadway	        5	    88	38	128
bike_lane		        normal      bike_lane	    1	    255	155	155
caution_zone	        stairs	    caution_zone	2	    255	192	0
caution_zone	        manhole	    caution_zone	2	    255	0	0
caution_zone	        tree_zone	caution_zone	2	    0	255	0
caution_zone	        grating	    caution_zone	2	    255	128	0
caution_zone	        repair_zone	caution_zone	2	    105	105	255

See more info at
modules/utils/surface_dataset_tools/surface_polygon.py
modules/utils/surface_dataset_tools/split_dataset.py
modules/dataloaders/datasets/surface.py
"""
labels = [
    'background',
    'bike_lane',
    'caution_zone',
    'crosswalk',
    'guide_block',
    'roadway',
    'sidewalk',
]
# RGB
colors = [
    [0, 0, 0],
    [255, 128, 0],
    [255, 0, 0],
    [255, 0, 255],
    [255, 255, 0],
    [0, 0, 255],
    [0, 255, 0],
]

num_classes = len(colors)  # 7
