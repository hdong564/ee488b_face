#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
from iresnet.models.iresnet import iresnet101
def MainModel(nOut=256, **kwargs):
    model = iresnet101(pretrained = False,num_classes = nOut, **kwargs)
    return model