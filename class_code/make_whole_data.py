import os
import cv2
import torch
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_size=(640, 480), device='cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_size = frame_size
        self.device = device
        self.frames = []
        self._load_frames()

    def _load_frames(self):
        video_paths = [os.path.join(self.root_dir, file) 
                       for file in os.listdir(self.root_dir) if file.endswith('.mp4')]

        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 초록색 채널만 추출
                green_channel = frame[:, :, 1]
                # 리사이즈
                resized_frame = cv2.resize(green_channel, self.frame_size)
                self.frames.append(resized_frame)

            cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        if self.transform:
            frame = self.transform(frame)  # Transform 적용
        else:
            frame = torch.tensor(frame, dtype=torch.float32, device=self.device) / 255.0  # 기본 정규화
            frame = frame.unsqueeze(0)  # (1, H, W) 형태로 변환

        return frame