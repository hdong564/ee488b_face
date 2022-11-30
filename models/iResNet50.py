#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
from iresnet.models.iresnet import iresnet50
def MainModel(nOut=256, **kwargs):
    model = iresnet50(pretrained = False,num_classes = nOut, **kwargs)
    return model