
from torchvision import models
import torch.nn as nn

def get_model():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = nn.Conv2d(256, 3, 1)
    return model
