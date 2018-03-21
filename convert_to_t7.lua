require 'loadcaffe'
require 'torch'
net = loadcaffe.load('vgg_face_caffe/VGG_FACE_deploy.prototxt', 'vgg_face_caffe/VGG_FACE.caffemodel')
torch.save('VGG_torch.t7', net)
