# -*- coding: utf-8 -*-
import torch
from torchvision import models


class ResNet18Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Model, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)


class ConvNextTinyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNextTinyModel, self).__init__()
        self.model = models.convnext_tiny(pretrained=True)
        self.model.classifier[2] = torch.nn.Linear(in_features=self.model.classifier[2].in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
