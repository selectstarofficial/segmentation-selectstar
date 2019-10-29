backbone = 'xception'
out_stride = 16  # network output stride (default: 8)
use_sbd = False  # whether to use SBD dataset
workers = 4
base_size = 513  # base image size
crop_size = 513  # crop image size

cuda = True
gpu_ids = [1,2,3]  # use which gpu to train
sync_bn = True if len(gpu_ids) > 1 else False  # whether to use sync bn
freeze_bn = False  # whether to freeze bn parameters (default: False)

epochs = 200
start_epoch = 0
batch_size = 4 * len(gpu_ids)
test_batch_size = 4 * len(gpu_ids)

loss_type = 'ce'  # 'ce': CrossEntropy, 'focal': Focal Loss
use_balanced_weights = False  # whether to use balanced weights (default: False)
lr = 1e-3
lr_scheduler = 'poly'  # lr scheduler mode: ['poly', 'step', 'cos']
momentum = 0.9
weight_decay = 5e-4
nesterov = False

resume = None  # put the path to resuming file if needed
checkname = "deeplab"  # set the checkpoint name

ft = False  # finetuning on a different dataset
eval_interval = 1  # evaluuation interval (default: 1)
no_val = False  # skip validation during training

dataset = 'coco'

root_dir = ''
if dataset == 'pascal':
    root_dir = '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
elif dataset == 'sbd':
    root_dir = '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
elif dataset == 'cityscapes':
    root_dir = '/path/to/datasets/cityscapes/'  # foler that contains leftImg8bit/
elif dataset == 'coco':
    root_dir = '/home/super/Projects/dataset/coco/'
else:
    print('Dataset {} not available.'.format(dataset))
    raise NotImplementedError
