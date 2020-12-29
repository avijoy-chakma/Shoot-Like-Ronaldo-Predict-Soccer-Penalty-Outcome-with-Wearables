import torch
from torch.autograd import Variable
import torch.nn as nn

def get_cls_loss(pred, gt):
    
    criterion = nn.CrossEntropyLoss()
    cls_loss = criterion(pred, gt)
#     cls_loss = F.nll_loss(F.log_softmax(pred, dim=0), gt)
    return cls_loss

