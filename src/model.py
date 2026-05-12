import torch
import torch.nn as nn
from torchvision import models


class FaceEmbeddingModel(nn.Module):

    def __init__(self):

        super(FaceEmbeddingModel, self).__init__()

        self.backbone = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
         )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(in_features, 128)

    def forward(self, x):

        embedding = self.backbone(x)

        return embedding