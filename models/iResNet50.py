#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import importlib

def MainModel(nOut=256, **kwargs):
    iresnet50 = importlib.import_module('models.iresnet.models.iresnet').__getattribute__('iresnet50')
    model = iresnet50(pretrained = False,num_classes = nOut)
    return model