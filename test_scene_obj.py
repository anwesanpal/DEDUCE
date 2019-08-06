# python test_scene_obj.py --dataset=places --thres=0.5

# Prediction for Scene+N_obj model
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

parser = argparse.ArgumentParser(description='DEDUCE Scene+N_obj Evaluation')
parser.add_argument('--cfg',default='yolov3/cfg/yolo_v3.cfg',help='config file for yolo')
parser.add_argument('--weight',default='yolov3/yolov3.weights',help='weight file for yolo')
parser.add_argument('--namesfile',default='yolov3/data/coco.names',help='name file for yolo')
parser.add_argument('--dataset',default='places',help='dataset to test')
parser.add_argument('--thres',default='0.5',type = float,help='confidence threshold for scene attributes')
parser.add_argument('--envtype',default='home',help='home or office type environment')

global args
args = parser.parse_args()

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = 'models/{}_best_{}.pth.tar'.format(arch, args.envtype)

model = models.__dict__[arch](num_classes=7)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365_{}.txt'.format(args.envtype)

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

cookbook = {}
cookbook["bathroom"] = ["sink","toilet"]
cookbook["office"] = ["tvmonitor","laptop","keyboard","mouse"]
cookbook["living_room"] = ["sofa","vase"]
cookbook["bedroom"] = ["bed"]
cookbook["dining_room"] = ["diningtable","chair","wine glass","bowl"]
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
        img = Image.open(img_dir)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        if(probs[0]>args.thres):
            pred = classes[idx[0]]
        else:
            objects, class_names = detect(args.cfg, args.weight, img_dir,args.namesfile)
            cnt = 0
            for i in range(len(cookbook)):
                if(bool(set(objects).intersection(cookbook.values()[i]))):
                    pred = cookbook.keys()[i]
                    cnt+=1
            if(cnt != 1):
                pred = classes[idx[0]]

        if(pred == class_name):
            correct+=1
        count+=1
    accuracy = 100*correct/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
    accuracies_list.append(accuracy)
print('Average test accuracy is = {:2.2f}%'.format(sum(accuracies_list)/len(accuracies_list)))