import os
import argparse


# Attribute Names
# Set Data Path (Important!)
data_root = '/home/work/PycharmProjects/data/CelebA/raw/'
with open(os.path.join(data_root, 'list_attr_celeba.txt'), 'r') as f:
    line = f.readlines()[1]
    attribute_names = sorted(line.split())


parser = argparse.ArgumentParser(description='Face Attributes (CelebA) Training')

parser.add_argument('-d', '--data-dir', metavar='DATADIR', default=data_root,
                    help='data directory')
parser.add_argument('-a', '--attr', metavar='ATTRIBUTE', default='Wearing_Necklace',
                    choices=attribute_names,
                    help='attribute to train: ' +
                        ' | '.join(attribute_names) +
                        ' (default: Eyeglasses)')
parser.add_argument('-c', '--num-classes', metavar='N', default=2,
                    help='number of classes (default: 2)')
# parser.add_argument('-m', '--model', metavar='MODEL', default='alexnet',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

args = parser.parse_args()



