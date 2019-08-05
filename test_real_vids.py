# python test_real_vids.py

# Prediction on real world videos
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
import cv2
from yolov3.detect import detect_vid
import time
from collections import Counter
import resnet

parser = argparse.ArgumentParser(description='DEDUCE real_world_video Evaluation')
parser.add_argument('--cfg',default='yolov3/cfg/yolo_v3.cfg',help='config file for yolo')
parser.add_argument('--weight',default='yolov3/yolov3.weights',help='weight file for yolo')
parser.add_argument('--namesfile',default='yolov3/data/coco.names',help='name file for yolo')
parser.add_argument('--envtype',default='home',help='home or office type environment')
parser.add_argument('--vid_in',default='',help='path to input video')
parser.add_argument('--vid_out',default='',help='path to output video')
parser.add_argument('--thres',default='0.5',type = float,help='confidence threshold for scene attributes')

global args
args = parser.parse_args()

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = 'models/%s_best.pth.tar' % arch
# if not os.access(model_file, os.W_OK):
#     weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
#     os.system('wget ' + weight_url)

if(args.envtype == 'home'):
    model = models.__dict__[arch](num_classes=7)
elif(args.envtype == 'office'):
    model = resnet.resnet18(num_classes=5)

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
# if not os.access(file_name, os.W_OK):
#     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
#     os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

total = 0

cookbook = {}
cookbook["bathroom"] = ["toilet"]
cookbook["office"] = ["laptop","keyboard","mouse"]
cookbook["living_room"] = ["sofa","vase"]
cookbook["bedroom"] = ["bed"]
cookbook["dining_room"] = ["diningtable","wine glass","bowl"]
cookbook["kitchen"] = ["oven","microwave","refrigerator"]

input_vid = args.vid_in
output_vid = args.vid_out

vidcap = cv2.VideoCapture(input_vid)
fps = vidcap.get(cv2.CAP_PROP_FPS)

t0 = time.time()
success,image = vidcap.read()
success = True

format = "XVID"
fourcc = cv2.VideoWriter_fourcc(*format)
video = cv2.VideoWriter(output_vid, fourcc, fps, (image.shape[1],image.shape[0]))
k = 0
last_k,labels = [],[]
text_to_put = ''
flag = False
while success:

    # You may need to convert the color.
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    input_img = V(centre_crop(im_pil).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    if(probs[0]>args.thres):
        text = classes[idx[0]]
    else:
        objects, class_names = detect_vid(args.cfg, args.weight, im_pil,args.namesfile)
        count = 0
        for i in range(len(cookbook)):
            if(bool(set(objects).intersection(cookbook.values()[i])) and (cookbook.keys()[i] in [classes[idx[0]],classes[idx[1]],classes[idx[2]]])):
                text = cookbook.keys()[i]
                count+=1
        if(count != 1):
            text = ''
    
    last_k.append(text)
    k+=1
    if(k==20):
        c = Counter(last_k)
        text_to_put = c.most_common(1)[0][0]
        if(text_to_put == ''):
            flag = True
        k = 0
        last_k = []

    labels.append(text_to_put)
    draw = ImageDraw.Draw(im_pil)
    font = ImageFont.truetype("abel-regular.ttf", 32)
    # if(text_to_put == '' and flag == True):
    #     text_to_put = labels[-2]
    #     flag = False
    draw.text((0, 0),text_to_put,(255,255,255),font=font)

    frame = cv2.cvtColor(np.asarray(im_pil), cv2.COLOR_RGB2BGR)
    video.write(frame)
    success,image = vidcap.read()
cv2.destroyAllWindows()
video.release()
print('Time taken is = {} seconds'.format(time.time()-t0))