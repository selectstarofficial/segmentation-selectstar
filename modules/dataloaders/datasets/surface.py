from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
import settings
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torchvision import transforms
from modules.dataloaders import custom_transforms as tr
from modules.dataloaders.utils import decode_segmap, encode_segmap

"""
base_dir
├── annotations
│   ├── *.xml
├── images
│   ├── *.jpg
├── masks
│   ├── *.png
├── train.txt
├── valid.txt
└── labels.xlsx
"""


class SurfaceSegmentation(Dataset):
    NUM_CLASSES = settings.num_classes

    def __init__(self, base_dir=settings.root_dir, split='train'):
        super().__init__()
        self.split = split
        self.images = []
        self.masks = []

        file_list = Path(base_dir) / f'{split}.txt'
        with open(file_list, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                image, mask = line.split(',')
                image = Path(base_dir) / image
                mask = Path(base_dir) / mask
                assert image.exists(), f'File not found: {image}'
                assert mask.exists(), f'File not found: {mask}'
                self.images.append(image)
                self.masks.append(mask)
        print(f'{split}| images: {len(self.images)}, masks: {len(self.masks)}')

        if split == 'train':
            self.composed_transform = transforms.Compose([
                tr.FixedResize(settings.resize_height, settings.resize_width),
                tr.RandomHorizontalFlip(),
                # tr.RandomRotate(90),
                # tr.RandomScaleCrop(base_size=settings.base_size, crop_size=settings.crop_size),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        elif split == 'valid':
            self.composed_transform = transforms.Compose([
                # tr.FixScaleCrop(crop_size=settings.crop_size),
                tr.FixedResize(settings.resize_height, settings.resize_width),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        else:
            raise KeyError("split must be one of 'train' or 'valid'.")

    def __getitem__(self, index):
        image_file, mask_file = self.images[index], self.masks[index]
        _img = self.preprocess_image(image_file)
        _mask = self.preprocess_mask(mask_file)
        sample = {'image': _img, 'label': _mask}

        return self.composed_transform(sample)

    def __len__(self):
        return len(self.masks)

    @staticmethod
    def preprocess_image(jpg_file):
        image = Image.open(jpg_file).convert('RGB')
        return image

    @staticmethod
    def preprocess_mask(png_file):
        image = np.array(Image.open(png_file), dtype=np.uint8)

        h, w, c = image.shape
        assert c == 3, f"Invalid channel number: {c}. {png_file}"

        new_mask = encode_segmap(image, dataset='surface')

        return Image.fromarray(new_mask)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dataset = SurfaceSegmentation(base_dir=settings.root_dir, split='train')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='surface')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
