from glob import glob
from pathlib import Path
import random
from os import path as osp

"""
By running this code train.txt, valid.txt will be generated.

base_dir
├── annotations
│   ├── *.xml
├── images
│   ├── *.jpg
├── masks
│   ├── *.png
├── train.txt <--
├── valid.txt <--
└── color_index.xlsx
"""

### RUN OPTIONS ###
BASE_DIR = Path('/home/super/Projects/dataset/surface6')
VALID_COUNT = 2000
###################

images = sorted(glob(str(BASE_DIR/'**/*.jpg'), recursive=True))
masks = sorted(glob(str(BASE_DIR/'**/*.png'), recursive=True))
images = [s for s in images if not '.ipynb_checkpoints' in s]
masks = [s for s in masks if not '.ipynb_checkpoints' in s]
print(len(images), len(masks))

masks = sorted(masks)
random.shuffle(masks)

train_lines = []
for mask in masks[VALID_COUNT:]:
    mask = Path(mask)
    name = mask.name.replace('.png', '')
    mask_file = f'masks/{name}.png'
    image_file = f'images/{name}.jpg'
    assert osp.exists(BASE_DIR / image_file), f"{image_file} not exists."

    line = f'{image_file},{mask_file}'
    train_lines.append(line)

with open(osp.join(BASE_DIR, 'train.txt'), 'w+') as f:
    f.write('\n'.join(train_lines))

valid_lines = []
for mask in masks[:VALID_COUNT]:
    mask = Path(mask)
    name = mask.name.replace('.png', '')
    mask_file = f'masks/{name}.png'
    image_file = f'images/{name}.jpg'
    assert osp.exists(BASE_DIR / image_file), f"{image_file} not exists."

    line = f'{image_file},{mask_file}'
    valid_lines.append(line)

with open(osp.join(BASE_DIR / 'valid.txt'), 'w+') as f:
    f.write('\n'.join(valid_lines))

print(f'train: {len(train_lines)}, valid: {len(valid_lines)}')
