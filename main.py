"""
Face Attributes (CelebA) Training
"""


import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchnet import meter  # 计量仪，记录训练过程中的统计数据
from torchnet import logger
from tqdm import tqdm

from dataset.celeba import CelebA
from utils.visualize import Visualizer

vis = Visualizer()


from utils import args


# Model Names
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

from models.alexnet import alexnet


def main():

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu)


def main_worker(gpu):

    args.gpu = gpu

    # create model
    if args.pretrained:
        print("=> using pre-trained model alexnet")
        model = alexnet(pretrained=True, num_classes=args.num_classes)
    else:
        print("=> creating model alexnet")
        model = alexnet(num_classes=args.num_classes)


    if args.gpu is not None:
        model.cuda()
        print("Use GPU: {} for training".format(args.gpu))

    # define loss function (criterion) and optimizer
    # criterion： 标准
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_set = CelebA(data_root=args.data_dir, attr=args.attr, split='train')
    val_set = CelebA(data_root=args.data_dir, attr=args.attr, split='val')

    train_loader = DataLoader(train_set,
                              args.batch_size,
                              drop_last=True,
                              shuffle=False,
                              num_workers=args.workers)
    val_loader = DataLoader(val_set,
                            args.batch_size,
                            drop_last=True,
                            shuffle=False,
                            num_workers=args.workers)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, val_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, args)

        model.save(epoch, args.epochs)


def train(train_loader, val_loader, model, criterion, optimizer, epoch):
    """
        在一个epoch上训练
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :return:
    """
    batch_time = meter.AverageValueMeter()
    data_time = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)

    meter_log = logger.MeterLogger()

    # switch to train mode
    model.train()

    end = time.time()

    # 遍历数据集训练
    pbar = tqdm(train_loader)
    ii = 0
    for (input, target) in pbar:

        pbar.set_description('Epoch[{:>2d}/{}] training on batches'.format(epoch, args.epochs))
        # measure data loading time
        data_time.add(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.data.item())

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

        if ii % args.print_freq == (args.print_freq - 1):
            vis.plot('loss', loss_meter.value()[0])
        ii += 1

    val_cm, val_accuracy = val(model, val_loader)

    vis.plot('val_accuracy', val_accuracy)
    print('pass')

    # print('Epoch[{:0>3d}/{}] : lr: *** - loss: {:.5f} - train_cm:{}, val_cm:{}'.format(
    #     epoch,
    #     loss_meter.value()[0],
    #     str(confusion_matrix.value()),
    #     str(float(val_cm.value()[0][0])),
    # ))


def val(model, dataloader):
    """
        cm stands for confusion_matrix
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()  # 将模型置于验证模式

    model.cuda()

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input)
        val_label = Variable(label.long())
        if args.gpu is not None:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        output = model(val_input)
        confusion_matrix.add(output.data.squeeze(), label.long())

    model.train()  ## 将模型恢复为训练模式

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    return confusion_matrix, accuracy


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

