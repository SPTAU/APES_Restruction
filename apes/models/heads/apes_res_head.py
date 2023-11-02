from typing import List, Union

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import nn


@MODELS.register_module()
class APESResHead(BaseModule):
    def __init__(self, init_cfg=None):
        super(APESResHead, self).__init__(init_cfg)
        self.conv1 = nn.Sequential(nn.Conv1d(2304, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Conv1d(128, 3, 1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)  # (B, 2304, 2048) -> (B, 256, 2048)
        x = self.dp1(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.dp2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv3(x)  # (B, 256, 2048) -> (B, 128, 2048)
        x = self.conv4(x)  # (B, 128, 2048) -> (B, 3, 2048)
        return x
