import numpy as np
from PIL import Image

import caffe
import glob, os

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('../../rgbd_benchmark/LabeledImages/bedroom_12_image.png')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)



for infile in glob.glob("../../rgbd_benchmark/LabeledImages/*_image.png"):
    im = Image.open(infile)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    im = Image.fromarray(out.astype(np.uint8), mode = 'P')
    print("Segmentation done for " + infile)
    im.save(infile + "_segmented.png")
