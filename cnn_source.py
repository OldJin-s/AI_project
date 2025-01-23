# 2. 모델 정의
import torch 
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import Sequential, GCNConv
import torch.optim as optimizer

class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 6, (3, 5, 5)),  # (T, H, W) -> (1, 6, 3, 5, 5)
            nn.ReLU(),
            nn.MaxPool3d(4, stride=3),  # (B, 6, 3, 2, 2)
            nn.Conv3d(6, 16, (3, 5, 5)),  # (B, 6, 3, 2, 2) -> (B, 16, 1, 1, 1)
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),  # (B, 16, 1, 1, 1)
        )
        self.flatten = nn.Flatten()

        # 최종 출력에 맞춰 fc 레이어 수정
        self.fc_layer = nn.Sequential(
            nn.Linear(126896, 120),  # (16) -> 출력 크기 (조정 필요)
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4)  # 클래스 수 3 (예: push_start, squat_start, lunge_start)
        )

    def forward(self, x):
        # Conv3d에 맞게 처리
        out = self.conv_layers(x)

        # 플래튼 후 FC 처리
        out = self.flatten(out)
        fc_out = self.fc_layer(out)
        
        return fc_out

