import torch
import cv2
import os
from torch.utils.data import Dataset

class PoseFrameDataset(Dataset):
    def __init__(self, wrong_dir, processed_dir, pose_type, transform=None, frame_size=(640, 480), device='cpu'):
        self.wrong_dir = wrong_dir
        self.processed_dir = processed_dir
        self.pose_type = pose_type
        self.transform = transform
        self.frame_size = frame_size
        self.device = device
        self.frame_pairs = []
        self._load_frame_pairs()

    def _load_frame_pairs(self):
        wrong_files = [
            f for f in os.listdir(self.wrong_dir) if f.startswith(self.pose_type) and f.endswith('.mp4')
        ]
        processed_files = [
            f for f in os.listdir(self.processed_dir) if f.startswith(self.pose_type) and f.endswith('.mp4')
        ]
        
        for wrong_file in wrong_files:
            file_number = wrong_file.split('_')[-1]
            corresponding_file = f"{self.pose_type}_graph_{file_number}"
            if corresponding_file in processed_files:
                wrong_path = os.path.join(self.wrong_dir, wrong_file)
                processed_path = os.path.join(self.processed_dir, corresponding_file)
                self.frame_pairs.append((wrong_path, processed_path))
            else:
                print(f"Warning: No corresponding processed file found for {wrong_file}")

    def __len__(self):
        return len(self.frame_pairs) * 2

    def _load_frames_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            green_channel = frame[:, :, 1]  # 초록색 채널 추출
            resized_frame = cv2.resize(green_channel, self.frame_size)  # 리사이즈
            frames.append(resized_frame)
        cap.release()
        return frames

    def _pad_or_crop_frames(self, frames, max_len):
        current_len = len(frames)
        if current_len < max_len:
            padding = torch.zeros((max_len - current_len, *frames[0].shape), dtype=torch.float32, device=self.device)
            frames = torch.cat([torch.tensor(frames, dtype=torch.float32, device=self.device), padding], dim=0)
        else:
            frames = torch.tensor(frames[:max_len], dtype=torch.float32, device=self.device)
        return frames

    def __getitem__(self, idx):
        pair_idx = 0
        cumulative_idx = 0

        for wrong_path, processed_path in self.frame_pairs:
            wrong_frames = self._load_frames_from_video(wrong_path)
            processed_frames = self._load_frames_from_video(processed_path)

            # 각 비디오에서 대응되는 최소 프레임 수만큼 처리
            num_frames = min(len(wrong_frames), len(processed_frames))

            if cumulative_idx + num_frames > idx:
                # 해당 프레임이 포함된 비디오를 찾음
                frame_idx = idx - cumulative_idx
                wrong_frame = wrong_frames[frame_idx]
                processed_frame = processed_frames[frame_idx]

                # 변환이 필요하면 적용
                if self.transform:
                    wrong_frame = self.transform(wrong_frame)
                    processed_frame = self.transform(processed_frame)

                # clone().detach()로 텐서 복사
                return wrong_frame.clone().detach(), processed_frame.clone().detach()

            cumulative_idx += num_frames

        # 만약 idx가 유효하지 않으면 빈 튜플 반환
        raise IndexError(f"Index {idx} is out of range for the dataset.")
