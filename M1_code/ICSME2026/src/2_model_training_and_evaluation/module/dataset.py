from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

default_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TrainDataset(Dataset):
    def __init__(self, data_list, dataset_dir, clip_len=31, transform=default_transform, bug_type="All_label", frame_skip=15):
        self.samples = []
        self.clip_len = clip_len
        self.half_len = clip_len // 2
        self.transform = transform
        self.bug_type = bug_type
        self.frame_skip = frame_skip

        dataset_dir = Path(dataset_dir)

        label_1_samples = []
        label_0_samples = []

        for data in data_list:
            frames_dir = dataset_dir / data['frames_path']
            frames_dir_name = frames_dir.name
            label_df = pd.read_csv(dataset_dir / data['frame_label_path'])
            label_df['frame_num'] = label_df['frame'].astype(str).str.replace('.jpg', '', regex=False).astype(int)

            frame_files = sorted(frames_dir.glob('*.jpg'), key=lambda x: int(x.stem))
            if not frame_files:
                continue
            frame_nums = [int(f.stem) for f in frame_files]
            min_frame = frame_nums[0]
            max_frame = frame_nums[-1]

            label_df['group'] = (label_df[self.bug_type] != label_df[self.bug_type].shift()).cumsum()
            
            bug_groups = label_df[label_df[self.bug_type] == 1].groupby('group')
            for _, group in bug_groups:
                group_frames = group['frame_num'].tolist()
                start_frame = group_frames[0]
                end_frame = group_frames[-1]
                
                candidates = list(range(start_frame, end_frame + 1, self.frame_skip))
                
                if candidates and candidates[0] == start_frame:
                    candidates = candidates[1:]
                    
                for center_idx in candidates:
                    if center_idx in group_frames:
                        sample_info = self._create_sample_info(
                            center_idx, min_frame, max_frame, frames_dir, 
                            1, data['data_name'], frames_dir_name
                        )
                        label_1_samples.append(sample_info)

            normal_frames = label_df[label_df[self.bug_type] == 0]['frame_num'].tolist()
            for center_idx in normal_frames:
                sample_info = self._create_sample_info(
                    center_idx, min_frame, max_frame, frames_dir, 
                    0, data['data_name'], frames_dir_name
                )
                label_0_samples.append(sample_info)

        num_bug_samples = len(label_1_samples)
        if num_bug_samples > 0 and len(label_0_samples) >= num_bug_samples:
            sampled_label_0 = random.sample(label_0_samples, num_bug_samples)
        else:
            sampled_label_0 = label_0_samples
            print(f"[Warning] Not enough label=0 samples ({len(label_0_samples)}) to match label=1 ({num_bug_samples})")

        self.samples = label_1_samples + sampled_label_0
        
        random.shuffle(self.samples)

    def _create_sample_info(self, center_idx, min_frame, max_frame, frames_dir, label, data_name, frames_dir_name):
        """フレームパスのリスト（パディング処理含む）とメタデータを辞書で返すヘルパー関数"""
        start = max(center_idx - self.half_len, min_frame)
        end   = min(center_idx + self.half_len, max_frame)
        indices = list(range(start, end + 1))

        if len(indices) < self.clip_len:
            pad_len = self.clip_len - len(indices)
            if start == min_frame:
                indices = [indices[0]] * pad_len + indices
            else:
                indices = indices + [indices[-1]] * pad_len

        frame_paths = [frames_dir / f"{idx}.jpg" for idx in indices]

        return {
            'frames': frame_paths,
            'label': label,
            'data_name': data_name,
            'frames_dir_name': frames_dir_name,
            'frame_name': f"{center_idx}.jpg"
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        for fp in sample['frames']:
            img = Image.open(fp).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        clip = torch.stack(frames).permute(1, 0, 2, 3)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return clip, label, sample['data_name'], sample['frames_dir_name'], sample['frame_name']

class TestDataset(Dataset):
    def __init__(self, data_list, dataset_dir, clip_len=31, transform=default_transform, bug_type="All_label"):
        self.samples = []
        self.clip_len = clip_len
        self.half_len = clip_len // 2
        self.transform = transform
        self.bug_type = bug_type

        dataset_dir = Path(dataset_dir)

        for data in data_list:
            frames_dir = dataset_dir / data['frames_path']
            frames_dir_name = frames_dir.name
            label_df = pd.read_csv(dataset_dir / data['frame_label_path'])
            label_df['frame_num'] = label_df['frame'].astype(str).str.replace('.jpg', '', regex=False).astype(int)

            frame_files = sorted(frames_dir.glob('*.jpg'), key=lambda x: int(x.stem))
            if not frame_files:
                continue
            frame_nums = [int(f.stem) for f in frame_files]

            for _, row in label_df.iterrows():
                center_idx = row['frame_num']

                start = max(center_idx - self.half_len, frame_nums[0])
                end   = min(center_idx + self.half_len, frame_nums[-1])
                indices = list(range(start, end + 1))

                if len(indices) < clip_len:
                    pad_len = clip_len - len(indices)
                    if start == frame_nums[0]:
                        indices = [indices[0]] * pad_len + indices
                    else:
                        indices = indices + [indices[-1]] * pad_len

                frame_paths = [frames_dir / f"{idx}.jpg" for idx in indices]

                self.samples.append({
                    'frames': frame_paths,
                    'label': row[self.bug_type],
                    'data_name': data['data_name'],
                    'frames_dir_name': frames_dir_name,
                    'frame_name': row['frame']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        for fp in sample['frames']:
            img = Image.open(fp).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        clip = torch.stack(frames).permute(1, 0, 2, 3)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return clip, label, sample['data_name'], sample['frames_dir_name'], sample['frame_name']

def make_data_list(situation_type, purpose, dataset_dir="./", data_split_csv="data_split.csv"):
    data_split_df = pd.read_csv(data_split_csv)
    data_list = []

    dataset_dir = Path(dataset_dir)

    for _, row in data_split_df[data_split_df[situation_type] == purpose].iterrows():
        data_name = row['Data_name']
        frame_data_dir = dataset_dir / data_name
        pairs = []

        idx = 1
        while True:
            f_dir = f"frames{idx}"
            f_label = f"frame_label{idx}.csv"
            
            if (frame_data_dir / f_dir).exists() and (frame_data_dir / f_label).exists():
                pairs.append((f_dir, f_label))
                idx += 1
            else:
                break

        for frames_path, frame_label_path in pairs:
            data_list.append({
                'frames_path': f"{data_name}/{frames_path}",
                'frame_label_path': f"{data_name}/{frame_label_path}",
                'data_name': data_name
            })

    return data_list
