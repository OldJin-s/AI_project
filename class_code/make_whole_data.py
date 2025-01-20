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

class PoseFrameDataset(Dataset):
    def __init__(self, wrong_dir, processed_dir, pose_type, transform=None, frame_size=(640, 480), device='cpu'):
        """
        잘못된 자세와 올바른 자세 데이터를 매칭하여 로드하는 데이터셋 클래스.
        
        Parameters:
            wrong_dir (str): 잘못된 자세 영상이 저장된 디렉토리 경로.
            processed_dir (str): 올바른 자세 영상이 저장된 디렉토리 경로.
            pose_type (str): 운동 유형 (예: 'squat', 'lunge', 'pushup').
            transform (callable, optional): 영상 전처리 함수.
            frame_size (tuple): 리사이즈할 영상 크기 (너비, 높이).
            device (str): 사용할 디바이스 ('cpu' 또는 'cuda' 또는 'mps').
        """
        self.wrong_dir = wrong_dir
        self.processed_dir = processed_dir
        self.pose_type = pose_type
        self.transform = transform
        self.frame_size = frame_size
        self.device = device
        self.frame_pairs = []  # (잘못된 자세, 올바른 자세) 쌍
        
        self._load_frame_pairs()

    def _load_frame_pairs(self):
        # 잘못된 자세 파일 목록
        wrong_files = [
            f for f in os.listdir(self.wrong_dir) if f.startswith(self.pose_type) and f.endswith('.mp4')
        ]
        
        # 올바른 자세 파일 목록
        processed_files = [
            f for f in os.listdir(self.processed_dir) if f.startswith(self.pose_type) and f.endswith('.mp4')
        ]
        
        # 매칭: 공통된 번호 기준
        for wrong_file in wrong_files:
            # 잘못된 자세 파일 번호 추출
            file_number = wrong_file.split('_')[-1]  # 예: "001.mp4"
            
            # 올바른 자세 파일 찾기
            corresponding_file = f"{self.pose_type}_graph_{file_number}"
            if corresponding_file in processed_files:
                wrong_path = os.path.join(self.wrong_dir, wrong_file)
                processed_path = os.path.join(self.processed_dir, corresponding_file)
                self.frame_pairs.append((wrong_path, processed_path))
            else:
                print(f"Warning: No corresponding processed file found for {wrong_file}")

    def __len__(self):
        # 데이터 구조 (wrong, processed)와 (processed, processed)의 두 가지 쌍을 처리
        return len(self.frame_pairs) * 2  # 각 쌍에 대해 두 가지 타입의 반환 값 처리

    def _load_frames_from_video(self, video_path):
        """
        주어진 비디오 경로에서 프레임을 로드하여 리스트로 반환합니다.
        """
        cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 초록색 채널만 추출 (필요시 수정 가능)
            green_channel = frame[:, :, 1]
            # 리사이즈
            resized_frame = cv2.resize(green_channel, self.frame_size)
            frames.append(resized_frame)
        cap.release()
        return frames

    def __getitem__(self, idx):
        # 데이터 구조 선택: (wrong_frames, processed_frames) 또는 (processed_frames, processed_frames)
        pair_idx = idx // 2
        structure_type = idx % 2  # 0: (wrong_frames, processed_frames), 1: (processed_frames, processed_frames)
        
        wrong_path, processed_path = self.frame_pairs[pair_idx]
        
        # 프레임 로드
        wrong_frames = self._load_frames_from_video(wrong_path)
        processed_frames = self._load_frames_from_video(processed_path)

        # 프레임을 Tensor로 변환 및 전처리
        wrong_frames = [
            self.transform(frame) if self.transform else torch.tensor(frame, dtype=torch.float32, device=self.device) / 255.0 
            for frame in wrong_frames
        ]
        processed_frames = [
            self.transform(frame) if self.transform else torch.tensor(frame, dtype=torch.float32, device=self.device) / 255.0 
            for frame in processed_frames
        ]

        if structure_type == 0:
            # (wrong_frames, processed_frames)
            return torch.stack(wrong_frames), torch.stack(processed_frames)
        else:
            # (processed_frames, processed_frames)
            return torch.stack(processed_frames), torch.stack(processed_frames)
