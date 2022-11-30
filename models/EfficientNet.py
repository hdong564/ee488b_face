from torchvision import models
import torch.nn as nn 
import torch.nn.functional as F
import torch
import timm
from timm.models.layers.classifier import ClassifierHead

def MainModel(nOut=256, **kwargs):
    return EfficientNet_b4(num_classes=nOut)


class EfficientNet_b4(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(EfficientNet_b4, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=pretrained)
        n_features = self.model.classifier.in_features

        self.fc = ClassifierHead(n_features, num_classes)
            
    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.fc(x)
        return x