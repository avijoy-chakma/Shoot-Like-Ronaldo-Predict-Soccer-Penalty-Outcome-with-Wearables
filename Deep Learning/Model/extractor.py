# encoding=utf-8
"""
    Created on 10:41 2019/07/07 
    @author: Avijoy Chakma
"""
import torch.nn as nn
import torch.nn.functional as F
# Input: 64,3,1,64

class AccExtractor(nn.Module):
    def __init__(self, win_size, first_channel, sec_channel, first_dim, sec_dim, out_dim, gt_size):
        self.win_size = win_size
        self.first_channel = first_channel
        self.sec_channel = sec_channel
        self.first_dim = first_dim
        self.sec_dim = sec_dim
        self.gt_size = gt_size
        self.out_dim = out_dim
            
        super(AccExtractor, self).__init__()
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.first_channel, kernel_size=(1, self.first_dim)),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.first_channel, out_channels=self.sec_channel, kernel_size=(1, self.sec_dim)),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.sec_channel*self.calculate_dimension(), out_features=self.out_dim),
            nn.Dropout(0.9)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=self.out_dim, out_features=self.gt_size)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out.reshape(-1, self.sec_channel * self.calculate_dimension())
        out = self.fc1(out)
        out = self.fc3(out)
        
        return out
    
    
    def calculate_dimension(self):
        first_layer_output = (((self.win_size - self.first_dim + 1)-2)/2)+1
        second_layer_output = (((first_layer_output - self.sec_dim + 1)-2)/2)+1
        return int(second_layer_output)