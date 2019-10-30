from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
import settings
from PIL import Image
import numpy as np

"""
base_dir
├── annotations*
├── images*
├── masks*
├── train.txt
├── valid.txt
└── color_index.xlsx
"""


class SurfaceSegmentation(Dataset):
    NUM_CLASSES = len(settings.class_names)

    def __init__(self, base_dir=settings.root_dir, split='train'):
        super().__init__()
        self.split = split
        self.train_files = []
        self.valid_files = []
        with open('train.txt', 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                image, mask = line.split(',')
                image = Path(base_dir)/'images'/image
                mask = Path(base_dir)/'masks'/mask
                assert image.exists(), f'File not found: {image}'
                assert mask.exists(), f'File not found: {mask}'
                self.train_files.append(image)
                self.valid_files.append(mask)
        print(f'train: {len(self.train_files)}, valid: {len(self.valid_files)}')

    def __getitem__(self, index):
        pass

    def preprocess_mask(self, png_file):
        image = Image.open(png_file)
        pass