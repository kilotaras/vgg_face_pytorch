#!/bin/bash

wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz

tar zxvf vgg_face_caffe.tar.gz
th convert_to_t7.lua
python convert_to_pytorch.py
python test.py

rm VGG_torch.t7
