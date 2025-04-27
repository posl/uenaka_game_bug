import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models.video import r3d_18
import torch.nn as nn

# デバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# コンテキストフレーム数
context_frames = 15

# データセットクラス
class BugDetection3DDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, context_frames=15):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.context_frames = context_frames

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # パディングする前後のフレーム数
        half = self.context_frames

        # 取得する範囲の開始・終了インデックス
        start = max(0, idx - half)
        end = min(len(self.data_frame) - 1, idx + half)

        indices = list(range(start, end + 1))  # 範囲内のインデックスをリストとして保持

        # もし必要なフレーム数が足りない場合、パディングする
        while len(indices) < 2 * half + 1:
            if start == 0:
                indices.insert(0, 0)  # 先頭が足りなければ、最初のフレームを追加
            elif end == len(self.data_frame) - 1:
                indices.append(len(self.data_frame) - 1)  # 末尾が足りなければ、最後のフレームを追加

        frame_list = []
        labels_list = []

        # それぞれのインデックスに対応する画像を取得
        for i in indices:
            img_name = os.path.join(self.root_dir, self.data_frame.iloc[i, 0])
            image = Image.open(img_name).convert('RGB')
            
            # 画像の前処理
            if self.transform:
                image = self.transform(image)

            frame_list.append(image)
            labels_list.append(int(self.data_frame.iloc[i, 1]))  # ラベルの取得

        # フレームのスタック（T,C,H,W）
        frames = torch.stack(frame_list).permute(1, 0, 2, 3)  # (C, T, H, W)
        
        # 現在のインデックスのラベルを返す
        label = torch.tensor(int(self.data_frame.iloc[idx, 1]))

        return frames, label

# 前処理
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# モデルの読み込みと構築
class ADDAFullModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_model = r3d_18()
        num_ftrs = base_model.fc.in_features  # ← 先に in_features を取得！
        base_model.fc = nn.Identity()         # ← それから fc を Identity に置き換える
        self.feature_extractor = base_model
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

def load_model(model_path):
    model = ADDAFullModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))  # map_location でCPU/GPU対応
    model = model.to(device)
    model.eval()
    return model

# 推論関数
def predict_and_save_to_csv(model, data_dir):
    with torch.no_grad():
        for sub_dir in os.listdir(data_dir):
            video_dir = os.path.join(data_dir, sub_dir)
            label_file = os.path.join(video_dir, 'frame_label.csv')
            frames_dir = os.path.join(video_dir, 'trimmed_frames')

            if os.path.isfile(os.path.join(video_dir, 'test')) and os.path.exists(label_file):
                print(f"Processing directory: {video_dir}")
                df = pd.read_csv(label_file)

                dataset = BugDetection3DDataset(csv_file=label_file, root_dir=frames_dir, transform=transform, context_frames=context_frames)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

                probabilities = []

                for frames in dataloader:
                    if isinstance(frames, list):  # ← これを追加
                        frames = frames[0]
                    frames = frames.to(device)
                    output = model(frames)
                    prob = torch.softmax(output, dim=1)[:, 1].item()
                    probabilities.append(prob)

                df['output'] = probabilities
                new_label_file = os.path.join(video_dir, 'frame_label_adda.csv')
                df.to_csv(new_label_file, index=False)
                print(f"Saved predictions to {new_label_file}")

# モデルパスとデータパスを指定
model_path = 'adda_model_full.pth'
dataset_path = '/local/uenaka/dataset2'

# 実行
model = load_model(model_path)
predict_and_save_to_csv(model, dataset_path)