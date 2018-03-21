import sys

import numpy as np
import torch
import torchvision.transforms as T

from skimage.io import imread
from torchvision import models

vgg_face = models.vgg16(pretrained=False, num_classes=2622)
vgg_face.eval()
chkp = torch.load('vgg_face.ptorch')
vgg_face.load_state_dict(chkp)

img = imread('vgg_face_caffe/ak.png')
means = np.array([129.1863,104.7624,93.5940])
img = img - means


tensor =T.ToTensor()(img)*255

perm = torch.LongTensor([2,1,0])
inp = tensor[perm].view(1,3,224,224)


inpv = torch.autograd.Variable(inp)

out = vgg_face(inpv)[0]

# manual softmax
x = out.data.numpy()
e_x = np.exp(x - np.max(x))
out = e_x / e_x.sum()

assert abs(out[2] - 0.9464)<1e-4

print("Test successfull")
