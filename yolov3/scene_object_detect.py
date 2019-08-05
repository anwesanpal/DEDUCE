#python scene_object_detect.py cfg/yolo_v3.cfg yolov3.weights data/coco.names

import os
import sys
import numpy as np
import time
from detect import detect

data_dir = os.path.join("/home/users/a2pal/Work/places365/data/places365_standard/")
mode = "train"
img_path = os.path.join(data_dir,mode)

classes  = ["bedroom","kitchen","bathroom","dining_room","office","living_room","home_office"]
cfgfile = sys.argv[1]
weightfile = sys.argv[2]
# imgfile = sys.argv[3]
# globals()["namesfile"] = sys.argv[3]
namesfile = sys.argv[3]
print(cfgfile,weightfile,namesfile)
t0 = time.time()
dict_scene = {}
for class_name in os.listdir(img_path):
    if(class_name not in classes):
        continue
    index = [0] * 80
    class_imgs = os.path.join(img_path,class_name)
    for img in os.listdir(class_imgs):
        print(os.path.join(class_imgs,img))
        objects, class_names = detect(cfgfile, weightfile, os.path.join(class_imgs,img),namesfile)
        exit()
        # print(objects,class_names)
        indices = [class_names.index(x) for x in objects]
        for i in indices:
            index[i]+=1 
    top_objind = sorted(range(len(index)), key=lambda i: index[i], reverse=True)[:10]
    top_obj = []
    for i in top_objind:
        top_obj.append(class_names[i])
    dict_scene[class_name] = top_obj
t1 = time.time()
print(dict_scene)
print("Time taken is {} secs".format(t1-t0))