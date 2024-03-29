import torch
import torchvision
import models


def resnet101(num_classes: int, inplace: bool, weights):
    model = torchvision.models.resnet101(
        num_classes=num_classes, weights=weights)
    model = torch.nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        torch.nn.Flatten(),
        model.fc,
    )
    if inplace is False:
        models.ReLU_inplace_to_False(model)
    return model
