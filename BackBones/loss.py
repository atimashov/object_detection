import torch
import torch.nn as nn

class SVMLoss(nn.Module):
    def __init__(self):
        super(SVMLoss, self).__init__()

    def forward(self, predictions, target):
        margin = 10
        b_size = predictions.size(0)
        corr_class_score = predictions[torch.arange(b_size), target].view(b_size, 1)
        loss = predictions - corr_class_score + margin
        mask = loss > 0
        loss = (loss * mask).sum(dim = 1) - margin
        return loss.mean()

class MyEntropy(nn.Module):
    def __init__(self):
        super(MyEntropy, self).__init__()

    def forward(self, predictions, target):
        b_size = predictions.size(0)
        lsm_func = nn.LogSoftmax(dim = 1)
        logsoftmax = lsm_func(predictions)
        loss = -logsoftmax[torch.arange(b_size), target]
        return loss.mean()