# encoding=utf-8
"""
    Created on 10:41 2018/11/10 
    @author: Avijoy Chakma
"""
import torch.nn as nn
import torch.nn.functional as F


class AccClassifier(nn.Module):
    def __init__(self, gt_size):
        super(AccClassifier, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128*12, out_features=32),
            nn.Dropout(0.9)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=gt_size),
        )

    def forward(self, x):
        out = x.reshape(-1, 128 * 12)
        out = self.fc1(out)
        out = self.fc3(out)
        return out
