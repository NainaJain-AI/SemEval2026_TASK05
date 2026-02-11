import torch.nn as nn

def get_loss():
    return nn.HuberLoss()
