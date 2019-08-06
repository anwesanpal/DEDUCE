# DEDUCE: Diverse scEne Detection methods in Unseen Challenging Environments

This is the implementation of our paper:

A. Pal, C. Nieto-Granda, and H. I. Christensen, “**DEDUCE: Diverse scEne Detection methods in Unseen Challenging Environments**,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Macau, China, Nov. 2019. [[arXiv](https://arxiv.org/pdf/1908.00191.pdf)]

## Prerequisite

1. The code has been implemented and tested on Ubuntu 16.04, python 2.7, PyTorch 0.4/1.1 (tested on NVIDIA Titan Xp with CUDA 9.0 and GeForce RTX 2080Ti with CUDA 10.0)
2. For the rest of dependencies, please run `pip install -r requirements.txt`
3. The pretrained models for scene recognition can be obtained [here](https://drive.google.com/open?id=1EVnOGJXBn4wo5V5eez4JsCxFs08fQUU_) and those for the yolov3 can be obtained [here](https://pjreddie.com/media/files/yolov3.weights). Put the scene models in a folder called "models", and place the yolo weights in the folder yolov3.

## Data

1. Places365: The official train/test splits can be found here. Please download and save it locally.
2. SUNRGBD: The data can be found [here](http://rgbd.cs.princeton.edu/data/SUNRGBD.zip). Please download and save it locally.
3. VPC: The 6 different home environments can be found [here](http://categorizingplaces.com/dataset.html). Please download and save it locally.

Update the locations accordingly in the config file.

## Training

For the Places and SUN dataset, please structure the data as `/path_to_data/dataset/train/scene_type/*.jpg`. For VPC dataset, please structure as `/path_to_data/dataset/train/scene_type/*.jpg`

## Evaluation on image datasets

1. Download the data for the 
