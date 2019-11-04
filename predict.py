# predict on jpg files or mp4 video

import cv2
import torch
from glob import glob
import os
import os.path as osp
from torchvision import transforms
from modules.dataloaders.utils import decode_segmap
from modules.models.deeplab_xception import DeepLabv3_plus
from modules.models.sync_batchnorm.replicate import patch_replication_callback
import numpy as np
from PIL import Image
from tqdm import tqdm

### RUN OPTIONS ###
MODEL_PATH = "./run/surface/deeplab/model_iou_77.pth.tar"
ORIGINAL_HEIGHT = 720
ORIGINAL_WIDTH = 1280
MODEL_HEIGHT = 512
MODEL_WIDTH = 1024
NUM_CLASSES = 7  # including background
CUDA = True if torch.cuda.is_available() else False

MODE = 'mp4'  # 'mp4' or 'jpg'
assert MODE in ['mp4', 'jpg'], "MODE should be 'mp4' or 'jpg'."
DATA_PATH = './input/test.mp4'  # .mp4 path or folder including *.jpg
OUTPUT_PATH = './output/output.mp4'  # where mp4 file or jpg frames folder should be saved.
SHOW_OUTPUT = False
######


class FrameGeneratorMP4:
    def __init__(self, mp4_file: str, output_path=None, show=True):
        assert osp.isfile(mp4_file), "DATA_PATH should be existing mp4 file path."
        self.vidcap = cv2.VideoCapture(mp4_file)
        self.fps = int(self.vidcap.get(cv2.CAP_PROP_FPS))
        self.total = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.show = show
        self.output_path = output_path

        if self.output_path is not None:
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.out = cv2.VideoWriter(OUTPUT_PATH, self.fourcc, self.fps, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))

    def __iter__(self):
        success, image = self.vidcap.read()
        for i in range(0, self.total):
            if success:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield np.array(img)

            success, image = self.vidcap.read()

    def __len__(self):
        return self.total

    def write(self, rgb_img):
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if self.show:
            cv2.imshow('output', bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                self.close()
                exit(1)

        if self.output_path is not None:
            self.out.write(bgr)

    def close(self):
        cv2.destroyAllWindows()
        self.vidcap.release()
        if self.output_path is not None:
            self.out.release()


class FrameGeneratorJpg:
    def __init__(self, jpg_folder: str, output_folder=None, show=True):
        assert osp.isdir(jpg_folder), "DATA_PATH should be directory including jpg files."
        self.files = sorted(glob(osp.join(jpg_folder, '*.jpg'), recursive=False))
        self.show = show
        self.output_folder = output_folder

        if self.output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            self.counter = 0

    def __iter__(self):
        for file in self.files:
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield np.array(img)

    def __len__(self):
        return len(self.files)

    def write(self, rgb_img):
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if self.show:
            cv2.imshow('output', bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                self.close()
                exit(1)

        if self.output_folder is not None:
            path = osp.join(self.output_folder, f'{str(self.counter).zfill(8)}.jpg')
            cv2.imwrite(path, bgr)

    def close(self):
        cv2.destroyAllWindows()


class ModelWrapper:
    def __init__(self):
        self.composed_transform = transforms.Compose([
            transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.model = self.load_model(MODEL_PATH)

    @staticmethod
    def load_model(model_path):
        model = DeepLabv3_plus(nInputChannels=3, n_classes=NUM_CLASSES, os=16)
        if CUDA:
            model = torch.nn.DataParallel(model, device_ids=[0])
            patch_replication_callback(model)
            model = model.cuda()
        if not osp.isfile(MODEL_PATH):
            raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        if CUDA:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch: {}, best_pred: {})"
              .format(model_path, checkpoint['epoch'], checkpoint['best_pred']))
        model.eval()
        return model

    def predict(self, rgb_img: np.array):
        x = self.composed_transform(Image.fromarray(rgb_img))
        x = x.unsqueeze(0)

        if CUDA:
            x = x.cuda()
        with torch.no_grad():
            output = self.model(x)
        pred = output.data.detach().cpu().numpy()
        pred = np.argmax(pred, axis=1).squeeze(0)
        segmap = decode_segmap(pred, dataset='surface')
        segmap = np.array(segmap * 255).astype(np.uint8)

        resized = cv2.resize(segmap, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
                             interpolation=cv2.INTER_NEAREST)
        return resized


def main():
    print('Loading model...')
    model_wrapper = ModelWrapper()

    if MODE == 'mp4':
        generator = FrameGeneratorMP4(DATA_PATH, OUTPUT_PATH, show=SHOW_OUTPUT)
    elif MODE == 'jpg':
        generator = FrameGeneratorJpg(DATA_PATH, OUTPUT_PATH, show=SHOW_OUTPUT)
    else:
        raise NotImplementedError('MODE should be "mp4" or "jpg".')

    for index, img in enumerate(tqdm(generator)):
        segmap = model_wrapper.predict(img)
        generator.write(segmap)

    generator.close()
    print('Done.')


if __name__ == '__main__':
    main()
