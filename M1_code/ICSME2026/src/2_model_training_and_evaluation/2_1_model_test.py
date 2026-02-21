from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.models.video import r3d_18
from torch.utils.data import DataLoader
from module.dataset import GameVideoDataset, make_data_list

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_SPLIT_CSV = DATA_DIR / "2_model_training_and_evaluation" / "data_split.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "models"

def evaluate_model(situation_type, model_game, test_game, bug_type):
    print(f"=== Start Evaluation: Model({model_game}) -> Test({test_game}) | {situation_type} | {bug_type} ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_list = make_data_list(
        situation_type=f"{situation_type}_{test_game}", 
        purpose="test", 
        dataset_dir=DATASET_DIR, 
        data_split_csv=DATA_SPLIT_CSV
    )
    
    test_dataset = GameVideoDataset(data_list=test_list, dataset_dir=DATASET_DIR, bug_type=bug_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_path = MODEL_SAVE_DIR / f"{bug_type}_{model_game}_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = r3d_18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    new_col_name = f"{situation_type}_{bug_type}_output"

    current_data = None
    current_frames_dir = None
    df = None
    fpath = None

    with torch.no_grad():
        for clips, labels, data_names, frames_dir_names, frame_names in tqdm(test_loader, desc="Testing"):
            clips = clips.to(device)
            outputs = model(clips)

            prob = torch.softmax(outputs, dim=1)[:, 1].cpu().item()

            data_name = data_names[0]
            frames_dir_name = frames_dir_names[0]
            
            frame_name = frame_names[0].item() if isinstance(frame_names[0], torch.Tensor) else frame_names[0]

            if (data_name != current_data) or (frames_dir_name != current_frames_dir):
                if df is not None and fpath is not None:
                    df.to_csv(fpath, index=False)

                current_data = data_name
                current_frames_dir = frames_dir_name

                csv_name = f"frame_label{frames_dir_name[-1]}.csv"

                fpath = DATASET_DIR / data_name / csv_name
                
                if fpath.exists():
                    df = pd.read_csv(fpath)
                    if new_col_name not in df.columns:
                        df[new_col_name] = np.nan
                else:
                    print(f"[Warning] CSV not found: {fpath}")
                    df = None

            if df is not None:
                df.loc[df['frame'] == int(frame_name), new_col_name] = prob

    if df is not None and fpath is not None:
        df.to_csv(fpath, index=False)

    print(f"=== Done! Test results saved with column name: {new_col_name} ===")

if __name__ == "__main__":
    experiments = [
        ("within-title", "SM64", "SM64", "All_label"),
        ("within-title", "SM64", "SM64", "OoB_label"),
        ("within-title", "SM64", "SM64", "SG_label"),
        ("within-title", "BotW", "BotW", "All_label"),
        ("within-title", "BotW", "BotW", "OoB_label"),
        ("within-title", "BotW", "BotW", "SG_label"),
        ("within-title", "BotW", "BotW", "FG_label"),
        ("within-title", "TotK", "TotK", "All_label"),
        ("within-title", "TotK", "TotK", "OoB_label"),
        ("within-title", "TotK", "TotK", "FG_label"),
        ("within-title", "TotK", "TotK", "IG_label"),
        ("cross-title", "SM64", "BotW", "All_label"),
        ("cross-title", "SM64", "BotW", "OoB_label"),
        ("cross-title", "SM64", "BotW", "SG_label"),
        ("cross-title", "BotW", "TotK", "All_label"),
        ("cross-title", "BotW", "TotK", "OoB_label"),
        ("cross-title", "BotW", "TotK", "FG_label")
    ]

    for situation_type, model_game, test_game, bug_type in experiments:
        evaluate_model(situation_type, model_game, test_game, bug_type)
