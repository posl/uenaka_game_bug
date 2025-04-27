import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r3d_18

# BugDetection3DDataset クラス
class BugDetection3DDataset(Dataset):
    def __init__(self, root_dir, transform=None, context_frames=15):
        self.frame_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))
        self.transform = transform
        self.context_frames = context_frames

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        half = self.context_frames
        start = max(0, idx - half)
        end = min(len(self.frame_paths), idx + half + 1)
        indices = list(range(start, end))
        while len(indices) < 2 * half + 1:
            if start == 0:
                indices.insert(0, 0)
            elif end == len(self.frame_paths):
                indices.append(len(self.frame_paths) - 1)

        frame_list = []
        for i in indices:
            image = Image.open(self.frame_paths[i]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frame_list.append(image)

        frames = torch.stack(frame_list)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        return frames


# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ソースモデルの特徴抽出器（fcをIdentityに）
source_feature_extractor = r3d_18()
num_ftrs = source_feature_extractor.fc.in_features
source_feature_extractor.fc = nn.Identity()
# strict=False で最後のfc層の不一致を無視して読み込む
source_feature_extractor.load_state_dict(torch.load("/local/uenaka/dataset2/3Dmodel_central_label.pth"), strict=False)
source_feature_extractor = source_feature_extractor.to(device)
source_feature_extractor.eval()

# ターゲット特徴抽出器
target_feature_extractor = r3d_18()
target_feature_extractor.fc = nn.Identity()
target_feature_extractor = target_feature_extractor.to(device)

# ドメイン判別器
discriminator = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, 1),
    nn.Sigmoid()
).to(device)

# ターゲットドメインデータ読み込み
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_train_dataset = BugDetection3DDataset(root_dir='frames', transform=transform)
target_train_loader = DataLoader(target_train_dataset, batch_size=32, shuffle=True)

# 損失関数とオプティマイザ
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(target_feature_extractor.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

# ADDA 学習ループ
epochs = 3
for epoch in range(epochs):
    target_feature_extractor.train()
    discriminator.train()

    epoch_d_loss = 0.0
    epoch_g_loss = 0.0

    # tqdmを使って進捗表示
    pbar = tqdm(target_train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
    for inputs in pbar:
        inputs = inputs.to(device)

        # ===== ソース特徴 =====
        with torch.no_grad():
            source_features = source_feature_extractor(inputs)

        # ===== ターゲット特徴 =====
        target_features = target_feature_extractor(inputs)

        # ===== ドメインラベル =====
        real_labels = torch.ones(inputs.size(0), 1).to(device)
        fake_labels = torch.zeros(inputs.size(0), 1).to(device)

        # ===== 判別器の学習 =====
        optimizer_D.zero_grad()
        d_loss_real = adversarial_loss(discriminator(source_features.detach()), real_labels)
        d_loss_fake = adversarial_loss(discriminator(target_features.detach()), fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # ===== ターゲット特徴抽出器の学習 =====
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(target_features), real_labels)
        g_loss.backward()
        optimizer_G.step()

        # ログの更新
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        pbar.set_postfix({"D_loss": d_loss.item(), "G_loss": g_loss.item()})

    print(f"[Epoch {epoch+1}/{epochs}] Avg D_loss: {epoch_d_loss/len(target_train_loader):.4f} | Avg G_loss: {epoch_g_loss/len(target_train_loader):.4f}")

# --- フルモデル定義 ---
class ADDAFullModel(nn.Module):
    def __init__(self, feature_extractor, num_ftrs, num_classes=2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# --- フルモデルインスタンス化と保存 ---
target_feature_extractor.eval()
full_model = ADDAFullModel(target_feature_extractor, num_ftrs=num_ftrs, num_classes=2)
torch.save(full_model.state_dict(), "adda_model_full.pth")
print("フルモデルを 'adda_model_full.pth' に保存しました。")