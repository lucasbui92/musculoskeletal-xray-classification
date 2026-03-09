import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
