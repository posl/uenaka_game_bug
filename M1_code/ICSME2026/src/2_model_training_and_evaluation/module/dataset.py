from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
import random

default_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class GameVideoDataset(Dataset):
    def __init__(self, data_list, dataset_dir="./", clip_len=31, transform=default_transform, bug_type="All_label", frame_skip=0):
        self.samples = []
        self.clip_len = clip_len
        self.transform = transform
        self.bug_type = bug_type
        self.frame_skip = frame_skip

        dataset_dir = Path(dataset_dir)

        for data in data_list:
            frames_dir = dataset_dir / data['frames_path']
            frames_dir_name = frames_dir.name
            
            label_df = pd.read_csv(dataset_dir / data['frame_label_path'])
            # .jpgを除去して数値化
            label_df['frame_num'] = label_df['frame'].astype(str).str.replace('.jpg', '', regex=False).astype(int)

            frame_files = sorted(frames_dir.glob('*.jpg'), key=lambda x: int(x.stem))
            frame_nums = [int(f.stem) for f in frame_files]

            if self.frame_skip > 0:
                center_candidates = frame_nums[::self.frame_skip + 1]
            else:
                center_candidates = frame_nums

            for _, row in label_df.iterrows():
                center_idx = row['frame_num']
                if center_idx not in center_candidates:
                    continue

                start = max(center_idx - clip_len // 2, frame_nums[0])
                end   = min(center_idx + clip_len // 2, frame_nums[-1])
                indices = list(range(start, end + 1))
                if len(indices) < clip_len:
                    if start == frame_nums[0]:
                        indices = indices + [indices[-1]] * (clip_len - len(indices))
                    else:
                        indices = [indices[0]] * (clip_len - len(indices)) + indices

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

class BalancedGameVideoDataset(GameVideoDataset):
    def __init__(self, data_list, dataset_dir="./", clip_len=31, transform=default_transform,
                 bug_type="All_label", frame_skip=0, balance=False):
        super().__init__(data_list, dataset_dir, clip_len, transform, bug_type, frame_skip)
        
        if balance:
            label_to_samples = defaultdict(list)
            for s in self.samples:
                label_to_samples[s['label']].append(s)

            min_count = min(len(label_to_samples[0]), len(label_to_samples[1]))
            balanced_samples = []
            for _, samples in label_to_samples.items():
                balanced_samples.extend(random.sample(samples, min_count))

            self.samples = balanced_samples

class GradCAMDataset(Dataset):
    def __init__(self, sample_list, root_dir='.', clip_len=31, transform=None):
        self.samples = []
        self.clip_len = clip_len
        self.half_len = clip_len // 2
        self.root_dir = Path(root_dir)
        self.transform = transform

        for rel_path in sample_list:
            rel_path = rel_path.strip()
            # str型のパスも想定して念のため Path 化
            full_path = self.root_dir / rel_path
            frame_dir = full_path.parent
            frame_num = int(full_path.stem)

            # pathlib の glob を使用
            frame_files = sorted(frame_dir.glob("*.jpg"), key=lambda x: int(x.stem))
            
            if not frame_files:
                print(f"[Warning] No images found in {frame_dir}")
                continue
                
            frame_nums = [int(f.stem) for f in frame_files]

            if frame_num not in frame_nums:
                print(f"[Warning] Target frame {frame_num} not found in {frame_dir}")
                continue

            start_idx = max(frame_num - self.half_len, frame_nums[0])
            end_idx = min(frame_num + self.half_len, frame_nums[-1])
            indices = list(range(start_idx, end_idx + 1))

            if len(indices) < clip_len:
                if start_idx == frame_nums[0]:
                    indices += [indices[-1]] * (clip_len - len(indices))
                else:
                    indices = [indices[0]] * (clip_len - len(indices)) + indices

            frame_paths = [frame_dir / f"{i}.jpg" for i in indices]

            self.samples.append({
                "frames": frame_paths,
                "center_path": rel_path,
                "data_name": frame_dir.parent.name,
                "frame_dir": frame_dir.name,
                "center_frame_num": frame_num
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []

        for fp in sample["frames"]:
            img = Image.open(fp).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        clip = torch.stack(frames).permute(1, 0, 2, 3)

        return (
            clip,
            sample["data_name"],
            sample["frame_dir"],
            sample["center_path"],
            sample["center_frame_num"]
        )

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
            
            # .exists() で存在確認
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
