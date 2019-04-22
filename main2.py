from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import os
import argparse

from models import *
from utils import progress_bar

from dataset import CelebA
from models import alexnet
from models import resnet18

from args import args

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


if __name__ == '__main__':
    # Data
    print('==> Preparing data ...')

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
    classes = ('yes', 'no')

    # Model
    print('==> Building model ...')
    # model = alexnet()
    model = resnet101()
    if hasattr(model, 'compile'):
        print('yes')
    model = model.to(device)
    # if device == 'cuda':
    #     model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True

    # Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    model.compile(lr=args.lr, epochs=args.epochs, criterion=criterion, optimizer=optimizer)

    model.load('/home/work/PycharmProjects/face_attr/checkpoints/models.resnet.ResNet_epoch[29.500]_190417114741.pt')
    # Train
    model.fit(train_loader=train_loader, val_loader=val_loader)

    model.save()
