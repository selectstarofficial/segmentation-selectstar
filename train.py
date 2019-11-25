import torch
import os
import numpy as np
from tqdm import tqdm

import settings
from modules.dataloaders import make_data_loader
from modules.models.sync_batchnorm.replicate import patch_replication_callback
from modules.models.deeplab_xception import DeepLabv3_plus, get_1x_lr_params, get_10x_lr_params
from modules.utils.loss import SegmentationLosses
from modules.utils.calculate_weights import calculate_weigths_labels
# from modules.utils.lr_scheduler import LR_Scheduler
from modules.utils.saver import Saver
from modules.utils.summaries import TensorboardSummary
from modules.utils.metrics import Evaluator

class Trainer(object):
    def __init__(self,):
        # Define Saver
        self.saver = Saver()
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': settings.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(**kwargs)

        # Define network
        model = DeepLabv3_plus(nInputChannels=3, n_classes=self.nclass, os=16, pretrained=settings.pretrained, _print=True)

        train_params = [{'params': get_1x_lr_params(model), 'lr': settings.lr},
                        {'params': get_10x_lr_params(model), 'lr': settings.lr}]

        # Define Optimizer
        # optimizer = torch.optim.SGD(train_params, momentum=settings.momentum,
        #                             weight_decay=settings.weight_decay, nesterov=settings.nesterov)
        optimizer = torch.optim.Adam(train_params)

        # Define Criterion
        # whether to use class balanced weights
        if settings.use_balanced_weights:
            classes_weights_path = os.path.join(settings.root_dir, settings.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(settings.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=settings.cuda).build_loss(mode=settings.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(settings.lr_scheduler, settings.lr,
        #                                     settings.epochs, len(self.train_loader))

        # Using cuda
        if settings.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=settings.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if settings.resume:
            if not os.path.isfile(settings.checkpoint):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(settings.checkpoint))
            checkpoint = torch.load(settings.checkpoint)
            settings.start_epoch = checkpoint['epoch']
            if settings.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not settings.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(settings.checkpoint, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if settings.ft:
            settings.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if settings.cuda:
                image, target = image.cuda(), target.cuda()
            # self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, settings.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * settings.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if settings.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
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
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * settings.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

if __name__ == "__main__":
    trainer = Trainer()
    print('Starting Epoch:', settings.start_epoch)
    print('Total Epoches:', settings.epochs)
    for epoch in range(settings.start_epoch, settings.epochs):
        trainer.training(epoch)
        if not settings.no_val and epoch % settings.eval_interval == (settings.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()
