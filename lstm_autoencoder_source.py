import torch
import torch.nn as nn

# CNN Autoencoder 모델 정의
class CNN_Autoencoder(nn.Module):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()
        
        # 인코더 (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 1채널 -> 16채널
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16채널 -> 32채널
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32채널 -> 64채널
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32채널 -> 64채널
            nn.ReLU()
        )
        
        # 디코더 (Decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64채널 -> 32채널
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64채널 -> 32채널
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32채널 -> 16채널
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16채널 -> 1채널 (원본 이미지로 복원)
            nn.Sigmoid()  # 0-1 범위로 출력되도록 Sigmoid 적용
        )
    
    def forward(self, x):
        # 인코더 부분
        x = self.encoder(x)
        
        # 디코더 부분
        x = self.decoder(x)
        
        return x

