"""CUDA_VISIBLE_DEVICES=1,2,3 python3 train.py"""

backbone = 'xception'
out_stride = 16  # network output stride (default: 8)
use_sbd = False  # whether to use SBD dataset
workers = 4
base_size = 513  # base image size
crop_size = 513  # crop image size

cuda = True

# If you want to use gpu:1,2,3, run CUDA_VISIBLE_DEVICES=1,2,3 python3 ...
# with gpu_ids option [0,1,2] starting with zero
gpu_ids = [0, 1, 2]  # use which gpu to train

sync_bn = True if len(gpu_ids) > 1 else False  # whether to use sync bn
freeze_bn = False  # whether to freeze bn parameters (default: False)

epochs = 200
start_epoch = 0
batch_size = 4 * len(gpu_ids)
test_batch_size = 4 * len(gpu_ids)

loss_type = 'ce'  # 'ce': CrossEntropy, 'focal': Focal Loss
use_balanced_weights = False  # whether to use balanced weights (default: False)
lr = 0.01
lr_scheduler = 'poly'  # lr scheduler mode: ['poly', 'step', 'cos']
momentum = 0.9
weight_decay = 5e-4
nesterov = False

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

### BGR -> name ###
color_to_class = {}
color_to_class[(255, 0, 0)] = 'sidewalk-block'
color_to_class[(217, 217, 217)] = 'sidewalk-cement'
color_to_class[(17, 89, 198)] = 'sidewalk-urethane'
color_to_class[(128, 128, 128)] = 'sidewalk-asphalt'
color_to_class[(153, 230, 255)] = 'sidewalk-soil_stone'
color_to_class[(35, 86, 55)] = 'sidewalk-damaged'
color_to_class[(70, 168, 110)] = 'sidewalk-other'
color_to_class[(0, 255, 255)] = 'braille_guilde_block-normal'
color_to_class[(0, 96, 128)] = 'braille_guilde_block-damaged'
color_to_class[(255, 170, 230)] = 'alley-normal'
color_to_class[(255, 88, 208)] = 'alley-crosswalk'
color_to_class[(200, 60, 138)] = 'alley-speed_bump'
color_to_class[(128, 38, 88)] = 'alley-damaged'
color_to_class[(155, 155, 255)] = 'bike_lane-normal'
color_to_class[(0, 192, 255)] = 'caution_zone-stairs'
color_to_class[(0, 0, 255)] = 'caution_zone-manhole'
color_to_class[(0, 255, 0)] = 'caution_zone-tree_zone'
color_to_class[(0, 128, 255)] = 'caution_zone-grating'
color_to_class[(255, 105, 105)] = 'caution_zone-repair_zone'

### id starting from 1 -> name ###
class_names = color_to_class.values()
id_to_class = {i + 1: name for i, name in enumerate(class_names)}
class_to_id = {value: key for key, value in id_to_class.items()}
