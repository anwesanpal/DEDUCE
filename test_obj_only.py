# python test_obj_only.py --dataset=places

# Prediction for Object_Only model
#
# by Anwesan Pal

import argparse
import numpy as np
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from yolov3.detect import detect
from config import places_dir, sun_dir

parser = argparse.ArgumentParser(description='DEDUCE Object_Only Evaluation')
parser.add_argument('--cfg',default='yolov3/cfg/yolo_v3.cfg',help='config file for yolo')
parser.add_argument('--weight',default='yolov3/yolov3.weights',help='weight file for yolo')
parser.add_argument('--namesfile',default='yolov3/data/coco.names',help='name file for yolo')
parser.add_argument('--dataset',default='places',help='dataset to test')
parser.add_argument('--envtype',default='home',help='home or office type environment')

global args
args = parser.parse_args()

cookbook = {}
cookbook["bathroom"] = ["sink","toilet"]
cookbook["office"] = ["tvmonitor","laptop","keyboard","mouse"]
cookbook["living_room"] = ["sofa","vase"]
cookbook["bedroom"] = ["bed"]
cookbook["dining_room"] = ["diningtable","wine glass","bowl", "chair"]
cookbook["kitchen"] = ["oven","microwave","refrigerator"]

if(args.dataset == 'places'):
    data_dir = places_dir + '/places365_standard_{}'.format(args.envtype)
    valdir = os.path.join(data_dir, 'val')
elif(args.dataset == 'sun'):
    data_dir = sun_dir
    valdir = os.path.join(data_dir, 'test')

accuracies_list = []
for class_name in os.listdir(valdir):
    correct,count=0,0
    for img_name in os.listdir(os.path.join(valdir,class_name)):
        img_dir = os.path.join(valdir,class_name,img_name)

        objects, class_names = detect(args.cfg, args.weight, img_dir,args.namesfile)
        for i in range(len(cookbook)):
            if(bool(set(objects).intersection(cookbook.values()[i]))):
                pred = cookbook.keys()[i]

        if(len(objects) == 0):
            pred = 'corridor'
        
        if(pred == class_name):
            correct+=1
        count+=1
    accuracy = 100*correct/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
    accuracies_list.append(accuracy)
print('Average test accuracy is = {:2.2f}%'.format(sum(accuracies_list)/len(accuracies_list)))