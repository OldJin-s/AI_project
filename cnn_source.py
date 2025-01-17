# 2. 모델 정의
import torch 
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import Sequential, GCNConv
import torch.optim as optimizer

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 6, 5,padding=2),  # 입력 채널 3 (RGB)
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),  # stride=2로 출력 크기 감소
            nn.Conv3d(6, 16, 5,padding = 2),  # padding=2로 출력 크기 유지
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),  # stride=2로 출력 크기 감소
        )
        self.flatten = nn.Flatten()
        # 계산된 출력 크기에 맞춰 fc_layer 수정
        self.fc_layer = nn.Sequential(
            nn.Linear(150528, 120),  # 수정된 입력 크기 (50176)
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 3) )

    def forward(self, x):
        out = self.conv_layers(x)
        flatten = self.flatten(out)
        fc_out = self.fc_layer(flatten)
        return fc_out