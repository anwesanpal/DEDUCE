# python compute_object_id.py ./data/places365_standard_slam/ -b=10 -j=4
# 
# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import resnet
import wideresnet
import pdb
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/users/a2pal/Work/pytorch-0.4-yolov3')
from detect import detect

parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='/home/users/a2pal/Work/places365/resnet18_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=7, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')
parser.add_argument('--cfg',default='yolo_files/cfg/yolo_v3.cfg',help='config file for yolo')
parser.add_argument('--weight',default='yolo_files/yolov3.weights',help='weight file for yolo')
parser.add_argument('--namesfile',default='yolo_files/data/coco.names',help='name file for yolo')

def main():
    global args, best_prec1
    args = parser.parse_args()
    print args

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path

    train_dataset = ImageFolderWithPaths(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, epoch)

        # evaluate on validation set
        validate(val_loader)

def get_hot_vector(objects,class_names):
    v = [0] * 80
    indices = [class_names.index(x) for x in objects]
    for i in indices:
        v[i]=1
    return v

def train(train_loader, epoch):
    for i, (input, target, path) in enumerate(train_loader):
        # measure data loading time
        obj_id_batch = []
        for j in range(len(path)):
            objects, class_names = detect(args.cfg, args.weight, path[j],args.namesfile)
            obj_hot_vector = get_hot_vector(objects, class_names)
            obj_id_batch.append(obj_hot_vector)

        # measure elapsed time

def validate(val_loader):
    for i, (input, target, path) in enumerate(val_loader):
        obj_id_batch = []
        for j in range(len(path)):
            objects, class_names = detect(args.cfg, args.weight, path[j],args.namesfile)
            obj_hot_vector = get_hot_vector(objects, class_names)
            obj_id_batch.append(obj_hot_vector)


if __name__ == '__main__':
    main()
