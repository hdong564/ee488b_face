#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
#from iresnet.models.iresnet import iresnet101
import importlib

def MainModel(nOut=256, **kwargs):
    iresnet101 = importlib.import_module('models.iresnet.models.iresnet').__getattribute__('iresnet101')
    model = iresnet101(pretrained = False,num_classes = nOut)
    return model