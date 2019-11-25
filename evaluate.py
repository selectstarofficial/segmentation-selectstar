import torch
import os
import numpy as np
from tqdm import tqdm

import settings
from modules.dataloaders import make_data_loader
from modules.models.sync_batchnorm.replicate import patch_replication_callback
from modules.models.deeplab_xception import DeepLabv3_plus
from modules.utils.loss import SegmentationLosses
from modules.utils.calculate_weights import calculate_weigths_labels
from modules.utils.metrics import Evaluator

"""
Running this program requires settings.py options below

settings.resume = True
settings.checkpoint = 'PathToCheckpointModel.pth.tar'
settings.dataset = 'surface'
settings.root_dir = '/path/to/surface6'
settings.num_classes
settings.resize_height
settings.resize_width
settings.batch_size
settings.workers
(settings.use_sbd=False)
settings.use_balanced_weights
settings.cuda
settings.loss_type
settings.gpu_ids
settings.labels
"""


class Trainer(object):
    def __init__(self, ):
        # Define Dataloader
        kwargs = {'num_workers': settings.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(**kwargs)

        # Define network
        self.model = DeepLabv3_plus(nInputChannels=3, n_classes=self.nclass, os=16, pretrained=settings.pretrained,
                               _print=True)

        # Define Criterion
        # whether to use class balanced weights
        if settings.use_balanced_weights:
            classes_weights_path = os.path.join(settings.root_dir, settings.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(settings.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=settings.cuda).build_loss(mode=settings.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if settings.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=settings.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if settings.resume is False:
            print("settings.resume is False but ignoring...")
        if not os.path.isfile(settings.checkpoint):
            raise RuntimeError("=> no checkpoint found at '{}'.\
            Please designate pretrained weights file to settings.checkpoint='~.pth.tar'.".format(settings.checkpoint))
        checkpoint = torch.load(settings.checkpoint)
        settings.start_epoch = checkpoint['epoch']
        if settings.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        # if not settings.ft:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(settings.checkpoint, checkpoint['epoch']))

    def validation(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if settings.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print('numImages: %5d' % (i * settings.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        ClassIoU = self.evaluator.Intersection_over_Union()

        print('IoU of each class')
        for index, label in enumerate(settings.labels):
            print('{}: {}'.format(label, ClassIoU[index]))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.validation()
