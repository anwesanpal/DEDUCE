# DEDUCE: Diverse scEne Detection methods in Unseen Challenging Environments

This is the implementation of our paper:

A. Pal, C. Nieto-Granda, and H. I. Christensen, “**DEDUCE: Diverse scEne Detection methods in Unseen Challenging Environments**,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Macau, China, Nov. 2019. [[arXiv](https://arxiv.org/pdf/1908.00191.pdf)]

## Prerequisite

1. The code has been implemented and tested on Ubuntu 16.04, python 2.7, PyTorch 0.4/1.1 (tested on NVIDIA Titan Xp with CUDA 9.0 and GeForce RTX 2080Ti with CUDA 10.0)
2. For the rest of dependencies, please run `pip install -r requirements.txt`
3. The pretrained models for scene recognition can be obtained [here(https://drive.google.com/open?id=1EVnOGJXBn4wo5V5eez4JsCxFs08fQUU_)] and those for the yolov3 can be obtained [here(https://pjreddie.com/media/files/yolov3.weights)]. Copy the scene models in a folder called models, and place the yolo weights in the folder yolov3.
