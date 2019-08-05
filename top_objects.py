# python top_objects.py --dataset=places --mode=val

# Outputs a dictionary giving the top 10 objects for every scene class from the Places or the SUN dataset
#
# by Anwesan Pal

import argparse
import numpy as np
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from yolov3.detect import detect
from collections import defaultdict
from config import places_dir, sun_dir

parser = argparse.ArgumentParser(description='PyTorch Places365 SLAM Evaluation')
parser.add_argument('--cfg',default='yolov3/cfg/yolo_v3.cfg',help='config file for yolo')
parser.add_argument('--weight',default='yolov3/yolov3.weights',help='weight file for yolo')
parser.add_argument('--namesfile',default='yolov3/data/coco.names',help='name file for yolo')
parser.add_argument('--dataset',default='places',help='dataset to test')
parser.add_argument('--mode',default='val',help='set to choose from')

global args
args = parser.parse_args()

if(args.dataset == 'places'):
    root_dir = places_dir + '/places365_standard_home'
    if(args.mode == 'train'):
        data_dir = os.path.join(root_dir, 'train')
    else:
        data_dir = os.path.join(root_dir, 'val')

elif(args.dataset == 'sun'):
    root_dir = sun_dir
    if(args.mode == 'train'):
        data_dir = os.path.join(root_dir, 'train')
    else:
        data_dir = os.path.join(root_dir, 'val')

def get_hot_vector(objects,class_names):
    v = [0] * 80
    indices = [class_names.index(x) for x in objects]
    for i in indices:
        v[i]=1
    return v

scene_dict = {}
scene_obj = defaultdict(list)
for class_name in os.listdir(data_dir):
    for img_name in os.listdir(os.path.join(data_dir,class_name)):
        img_dir = os.path.join(data_dir,class_name,img_name)

        objects, class_names = detect(args.cfg, args.weight, img_dir,args.namesfile)
        obj_hot_vector = get_hot_vector(objects, class_names)

        if class_name not in scene_dict:
            scene_dict[class_name] = obj_hot_vector
        else:
            scene_dict[class_name] = map(lambda x,y:x+y, scene_dict[class_name], obj_hot_vector)

    indices = sorted(range(len(scene_dict[class_name])), key=lambda i: scene_dict[class_name][i], reverse=True)[:10]
    for ind in indices:
        scene_obj[class_name].append(class_names[ind])

print(dict(scene_obj))