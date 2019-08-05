# python test_combined.py --dataset=places --envtype=home
# python test_combined.py --dataset=vpc --hometype=home1 --floortype=data_1

# Prediction for Combined model
#
# by Anwesan Pal

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import resnet
from yolov3.detect import detect
from config import places_dir, sun_dir, vpc_dir
from train_deduce_combined import get_hot_vector

parser = argparse.ArgumentParser(description='DEDUCE Combined Evaluation')
parser.add_argument('--cfg',default='yolov3/cfg/yolo_v3.cfg',help='config file for yolo')
parser.add_argument('--weight',default='yolov3/yolov3.weights',help='weight file for yolo')
parser.add_argument('--namesfile',default='yolov3/data/coco.names',help='name file for yolo')
parser.add_argument('--dataset',default='places',help='dataset to test')
parser.add_argument('--hometype',default='home1',help='home type to test')
parser.add_argument('--floortype',default='data_0',help='data type to test')
parser.add_argument('--envtype',default='home',help='home or office type environment')

global args
args = parser.parse_args()

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = 'models/%s_best_slam.pth.tar' % arch

# if not os.access(model_file, os.W_OK):
#     weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
#     os.system('wget ' + weight_url)

class Object_Linear(nn.Module):
    def __init__(self):
        super(Object_Linear, self).__init__()
        self.fc = nn.Linear(80, 512)

    def forward(self, x):
        out = self.fc(x)
        return out
object_idt = Object_Linear()

class LinClassifier(nn.Module):
    def __init__(self,num_classes):
        super(LinClassifier, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, conv, idt):
        out = torch.cat((conv,idt),1)
        out = self.fc(out)
        return out
classifier = LinClassifier(7)

# model = models.__dict__[arch](num_classes=7)
model = resnet.resnet18(num_classes = 7)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
model_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['model_state_dict'].items()}
obj_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['obj_state_dict'].items()}
classifier_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['classifier_state_dict'].items()}
model.load_state_dict(model_state_dict)
object_idt.load_state_dict(obj_state_dict)
classifier.load_state_dict(classifier_state_dict)
model.eval()
object_idt.eval()
classifier.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365_{}.txt'.format(args.envtype)
# if not os.access(file_name, os.W_OK):
#     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
#     os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

if(args.dataset == 'places'):
    data_dir = places_dir + '/places365_standard_{}'.format(args.envtype)
    valdir = os.path.join(data_dir, 'val')
elif(args.dataset == 'sun'):
    data_dir = sun_dir
    valdir = os.path.join(data_dir, 'test')
elif(args.dataset == 'vpc'):
    data_dir = vpc_dir
    home_dir = os.path.join(data_dir, 'data_'+args.hometype)
    valdir = os.path.join(home_dir,args.floortype)

accuracies_list = []
for class_name in os.listdir(valdir):
    correct,count=0,0
    for img_name in os.listdir(os.path.join(valdir,class_name)):
        img_dir = os.path.join(valdir,class_name,img_name)
        img = Image.open(img_dir)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        output_conv = model.forward(input_img)
        objects, class_names = detect(args.cfg, args.weight, img_dir,args.namesfile)
        obj_hot_vector = get_hot_vector(objects, class_names)
        t = torch.autograd.Variable(torch.FloatTensor(obj_hot_vector))
        output_idt = object_idt(t)
        output_idt = output_idt.unsqueeze(0)
        logit = classifier(output_conv,output_idt)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        if(classes[idx[0]] == class_name):
            correct+=1
        count+=1
    accuracy = 100*correct/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
    accuracies_list.append(accuracy)
print('Average test accuracy is = {:2.2f}%'.format(sum(accuracies_list)/len(accuracies_list)))