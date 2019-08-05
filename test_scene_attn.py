# python test_scene_attn.py --dataset=places

# Prediction for Scene+attn model
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
from scipy.misc import imresize as imresize
import cv2
from config import places_dir, sun_dir, vpc_dir

parser = argparse.ArgumentParser(description='DEDUCE Scene+attn Evaluation')
parser.add_argument('--dataset',default='places',help='dataset to test')
parser.add_argument('--envtype',default='home',help='home or office type environment')

global args
args = parser.parse_args()

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365_{}.txt'.format(args.envtype)
    # if not os.access(file_name_category, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    #     os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    # if not os.access(file_name_IO, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
    #     os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tf = trn.Compose([
            trn.Scale(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            normalize,
        ])
    return tf

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

        def load_model():
            # this model has a last conv feature map as 14x14

            model_file = 'models/wideresnet_best.pth.tar'
            # if not os.access(model_file, os.W_OK):
            #     os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
            #     os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

            import wideresnet
            model = wideresnet.resnet18(num_classes=7)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.eval()

            # the following is deprecated, everything is migrated to python36

            ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
            #from functools import partial
            #import pickle
            #pickle.load = partial(pickle.load, encoding="latin1")
            #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

            # hook the feature extractor
            features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
            for name in features_names:
                model._modules.get(name).register_forward_hook(hook_feature)
            return model


        # load the labels
        classes, labels_IO, labels_attribute, W_attribute = load_labels()

        # load the model
        features_blobs = []
        model = load_model()

        # load the transformer
        tf = returnTF() # image transformer

        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = params[-2].data.numpy()
        weight_softmax[weight_softmax<0] = 0

        img_dir = os.path.join(valdir,class_name,img_name)
        img = Image.open(img_dir)
        input_img = V(tf(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        # idx = idx.numpy()
        
        # # generate class activation mapping
        # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # # render the CAM and output
        # img = cv2.imread(img_dir)
        # height, width, _ = img.shape
        # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.4 + img * 0.6
        # correct_dir = os.path.join(valdir_attn,class_name,img_name)
        # wrong_dir = os.path.join(valdir_attn,class_name,'wrong_attn/',img_name+'_'+classes[idx[0]])

        if(classes[idx[0]] == class_name):
            correct+=1
        count+=1
            # cv2.imwrite(correct_dir, result)
        # else:
            # cv2.imwrite(wrong_dir, result)
    accuracy = 100*correct/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
    accuracies_list.append(accuracy)
print('Average test accuracy is = {:2.2f}%'.format(sum(accuracies_list)/len(accuracies_list)))