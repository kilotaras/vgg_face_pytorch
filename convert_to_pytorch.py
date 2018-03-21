import sys
import torch

from torch.utils.serialization import load_lua
from torchvision import models

net = load_lua('VGG_torch.t7')

vgg16 = models.vgg16(pretrained=False, num_classes=2622)

def short(f):
    return f.__class__.__name__

def copyRelu(f, t):
    assert short(t) == 'ReLU'

def copySpatial(f, t):
    assert short(t) == 'Conv2d'
    assert t.kernel_size == (f.kH, f.kW)
    assert t.stride == (f.dH, f.dW)
    assert t.padding == (f.padH, f.padW)
    assert t.in_channels == f.nInputPlane
    assert t.out_channels == f.nOutputPlane

    t.bias.data = f.bias.type('torch.FloatTensor')
    t.weight.data = f.weight.type('torch.FloatTensor')

def copyPool(f,t):
    assert short(t) == 'MaxPool2d'
    assert t.kernel_size == f.kH
    assert t.kernel_size == f.kW
    assert t.stride == f.dH
    assert t.stride == f.dW
    assert t.padding == f.padH
    assert t.padding == f.padW

def copyDropout(f, t):
    assert short(t) == 'Dropout'

def copyLinear(f, t):
    assert short(t) == 'Linear'
    assert f.weight.size() == t.weight.size()
    t.weight.data = f.weight.type('torch.FloatTensor')
    t.bias.data = f.bias.type('torch.FloatTensor')

def copy(f, t):
    name = short(f)
    if name == 'ReLU':
        copyRelu(f,t)
    elif name == 'SpatialConvolution':
        copySpatial(f, t)
    elif name == 'SpatialMaxPooling':
        copyPool(f,t)
    elif name == 'Dropout':
        copyDropout(f,t)
    elif name == 'Linear':
        copyLinear(f,t)
    else:
        assert False, name

features = vgg16.features

for ind, layer in enumerate(features):
    copy(net.get(ind), layer)

clf = vgg16.classifier

for ind, layer in enumerate(clf):
    copy(net.get(ind+32), layer)

torch.save(vgg16.state_dict(), 'vgg_face.ptorch')
