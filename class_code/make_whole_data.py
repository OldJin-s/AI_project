import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2
import pickle
class WholeExerciseDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_video=90, device='cpu'):
        """
        :param root_dir: 데이터 폴더의 최상위 디렉토리 (예: "data/")
        :param transform: 영상 데이터에 적용할 변환 함수
        :param frames_per_video: 영상에서 추출할 프레임 수
        :param device: 데이터를 로드할 디바이스 ('cpu' 또는 'cuda')
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.device = device

        # 모든 영상 파일의 경로만 저장
        self.video_paths = []

        for file in os.listdir(root_dir):
            if file.endswith('.mp4'):  # MP4 파일만 처리
                self.video_paths.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # 영상 읽기 및 프레임 추출
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.frames_per_video:
            ret, frame = cap.read()
            if not ret:  # 영상이 끝났으면 루프 중단
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 읽으므로 RGB로 변환
            frames.append(frame)

        cap.release()

        # 프레임 수가 부족할 경우 반복하여 채움
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])  # 마지막 프레임을 반복해서 추가

        # 텐서 변환 및 전처리
        frames = torch.tensor(frames, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)  # (T, H, W, C) → (T, C, H, W)
        if self.transform:
            frames = self.transform(frames)

        return frames  # 라벨 없이 영상만 반환