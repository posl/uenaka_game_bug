from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from tqdm import tqdm
from module.dataset import TrainDataset, make_data_list

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_SPLIT_CSV = DATA_DIR / "2_model_training_and_evaluation" / "data_split.csv"

MODEL_SAVE_DIR = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def train_model(game, bug_type, batch_size=32, num_epochs=3, num_frame_skip=4, lr=0.001, momentum=0.9):
    print(f"=== Start Training: {game} | {bug_type} ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_list = make_data_list(situation_type=f"within-title_{game}", purpose="train", dataset_dir=DATASET_DIR, data_split_csv=DATA_SPLIT_CSV)
    val_list   = make_data_list(situation_type=f"within-title_{game}", purpose="validation", dataset_dir=DATASET_DIR, data_split_csv=DATA_SPLIT_CSV)

    train_dataset = TrainDataset(data_list=train_list, dataset_dir=DATASET_DIR, bug_type=bug_type, frame_skip=0)
    val_dataset   = TrainDataset(data_list=val_list, dataset_dir=DATASET_DIR, bug_type=bug_type, frame_skip=num_frame_skip)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = r3d_18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for clips, labels, _, _, _ in train_pbar:
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for clips, labels, _, _, _ in val_pbar:
                clips = clips.to(device)
                labels = labels.to(device)

                outputs = model(clips)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    save_path = MODEL_SAVE_DIR / f"{bug_type}_{game}_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model

if __name__ == "__main__":
    experiments = [
        ("SM64", "All_label"),
        ("SM64", "OoB_label"),
        ("SM64", "SG_label"),
        ("BotW", "All_label"),
        ("BotW", "OoB_label"),
        ("BotW", "SG_label"),
        ("BotW", "FG_label"),
        ("TotK", "All_label"),
        ("TotK", "OoB_label"),
        ("TotK", "FG_label"),
        ("TotK", "IG_label"),
    ]
    for game, bug_type in experiments:
        train_model(game=game, bug_type=bug_type)
