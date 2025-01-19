import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import pickle


class ExerciseDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_video=16, device='cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.device = device

        self.classes = {
            "push_start": [1, 0, 0, 0],
            "squat_start": [0, 1, 0, 0],
            "lunge_start": [0, 0, 1, 0],
            "not_start": [0, 0, 0, 1]
        }

        self.video_paths = []
        self.labels = []

        for class_name, label in self.classes.items():
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                for file in os.listdir(class_folder):
                    if file.endswith('.mp4'):
                        self.video_paths.append(os.path.join(class_folder, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32, device=self.device)

        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            green_frame = frame[:,:,1]  # 초록색 채널만 추출
            frames.append(green_frame)  # (H, W) 형태로 저장

        cap.release()

        # 부족한 프레임을 0으로 채움
        while len(frames) < self.frames_per_video:
            frames.append(np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=np.float32))  # (H, W) 크기의 0으로 채움

        # 텐서로 변환하고 permute (T, C, H, W)
        frames = torch.tensor(np.array(frames), dtype=torch.float32, device=self.device).unsqueeze(1).permute(0, 1, 2, 3)  # (T, C, H, W)

        # transform을 각 프레임에 대해 개별적으로 적용
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])  # 각 프레임에 transform 적용

        return frames, label