from torch import nn


def ReLU_inplace_to_False(module: nn.Module):
    for layer in module.children():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)
