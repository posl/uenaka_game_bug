from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from module.dataset import GradCAMDataset
from module.gradcam import GradCAM3D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_SPLIT_CSV = DATA_DIR / "2_model_training_and_evaluation" / "data_split.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "models"
PR_LIST_DIR = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "PR_list"
OUTPUT_DIR = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "grad_cam_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_gradcam(situation_type, test_game, bug_type, num_samples=5, min_frame_dist=1800):
    print(f"=== Starting Grad-CAM for {situation_type} | {test_game} | {bug_type} ===")

    def select_sample_with_threshold():
        print(f"--- Sampling for {situation_type} {test_game} ({bug_type}) ---")
        
        pr_csv_path = PR_LIST_DIR / situation_type / test_game / f"{bug_type}.csv"
        if not pr_csv_path.exists():
            raise FileNotFoundError(f"PR curve data not found: {pr_csv_path}")
        
        df_pr = pd.read_csv(pr_csv_path)
        
        precision = df_pr["precision"].values
        recall = df_pr["recall"].values
        thresholds = df_pr["threshold"].values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        
        if np.isinf(best_threshold) or np.isnan(best_threshold):
            best_threshold = 0.5
            
        print(f"Optimal Threshold (Max F1): {best_threshold:.4f}")

        df_purpose = pd.read_csv(DATA_SPLIT_CSV)
        test_names = df_purpose.loc[
            df_purpose[f"{situation_type}_{test_game}"] == "test", "Data_name"
        ].tolist()

        candidates_tp = []
        candidates_fp = []
        candidates_fn = []

        bug_id_map = {}

        output_col = f"{situation_type}_{bug_type}_output"
        label_col = f"{bug_type}"

        for name in test_names:
            target_data_dir = DATASET_DIR / name
            if not target_data_dir.exists():
                continue
                
            # pathlib で csv 検索
            for path in target_data_dir.glob("*.csv"):
                df = pd.read_csv(path)

                if output_col not in df.columns or label_col not in df.columns or "frame" not in df.columns:
                    continue

                outputs = df[output_col].values
                labels = df[label_col].values
                frames = df["frame"].values
                
                bug_counter = 0
                is_in_bug = False
                
                csv_filename = path.name
                
                for f, l in zip(frames, labels):
                    if l == 1:
                        if not is_in_bug:
                            bug_counter += 1
                            is_in_bug = True
                        bug_id_map[f"{name}_{csv_filename}_{f}"] = f"{name}_{csv_filename}_bug_{bug_counter}"
                    else:
                        is_in_bug = False

                for f, o, l in zip(frames, outputs, labels):
                    suffix = csv_filename.replace('frame_label', '').replace('.csv', '')
                    frame_path = f"{name}/frames{suffix}/{f}.jpg"
                    current_bug_id = bug_id_map.get(f"{name}_{csv_filename}_{f}", None)

                    item = {
                        "path": frame_path,
                        "video": name,
                        "frame": f,
                        "score": o,
                        "bug_id": current_bug_id
                    }

                    if l == 1 and o >= best_threshold:
                        candidates_tp.append(item)
                    elif l == 0 and o >= best_threshold:
                        candidates_fp.append(item)
                    elif l == 1 and o < best_threshold:
                        candidates_fn.append(item)

        def select_diverse_top_k(candidates, k, sort_descending=True, is_bug_label=False):
            candidates.sort(key=lambda x: x["score"], reverse=sort_descending)
            
            selected = []
            used_bug_ids = set()
            
            for cand in candidates:
                if len(selected) >= k:
                    break
                
                is_duplicate = False
                
                if is_bug_label:
                    bid = cand["bug_id"]
                    if bid is not None and bid in used_bug_ids:
                        is_duplicate = True
                else:
                    for s in selected:
                        cand_frame_int = int(str(cand["frame"]).replace('.jpg', ''))
                        s_frame_int = int(str(s["frame"]).replace('.jpg', ''))
                        if cand["video"] == s["video"] and abs(cand_frame_int - s_frame_int) < min_frame_dist:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    selected.append(cand)
                    if cand["bug_id"] is not None:
                        used_bug_ids.add(cand["bug_id"])
            
            return [s["path"] for s in selected]

        final_tp = select_diverse_top_k(candidates_tp, num_samples, sort_descending=True, is_bug_label=True)
        final_fp = select_diverse_top_k(candidates_fp, num_samples, sort_descending=True, is_bug_label=False)
        final_fn = select_diverse_top_k(candidates_fn, num_samples, sort_descending=False, is_bug_label=True)

        print(f"Selected: TP={len(final_tp)}, FP={len(final_fp)}, FN={len(final_fn)}")
        return final_tp, final_fp, final_fn

    TP_sample, FP_sample, FN_sample = select_sample_with_threshold()
    samples = [TP_sample, FP_sample, FN_sample]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = r3d_18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)

    model_path = MODEL_SAVE_DIR / f"{bug_type}_{test_game}_model.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"[Warning] Model file {model_path} not found. Using untrained weights.")

    model = model.to(device)
    model.eval()

    gradcam = GradCAM3D(model, target_layer="layer4")

    default_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for sample, result_type_full in zip(samples, ["TP_sample", "FP_sample", "FN_sample"]):
        result_type = result_type_full.replace("_sample", "")

        if not sample:
            print(f"No samples for {result_type}")
            continue

        test_dataset = GradCAMDataset(sample, root_dir=DATASET_DIR, transform=default_transform)
        test_loader = DataLoader(test_dataset, batch_size=1)

        for clip, data_name, frame_dir, center_path, center_num in tqdm(test_loader, desc=f"{result_type}"):
            clip = clip.to(device)
            clip.requires_grad = True

            try:
                cam = gradcam(clip)
            except RuntimeError as e:
                print(f"GradCAM error: {e}")
                continue

            if isinstance(cam, torch.Tensor):
                cam = cam.detach().cpu().numpy()
            cam = np.squeeze(cam)
            clip_np = clip.detach().cpu().squeeze().permute(1, 2, 3, 0).numpy()

            vis_frames = []
            for t in range(cam.shape[0]):
                frame = (clip_np[t] - clip_np[t].min()) / (clip_np[t].max() - clip_np[t].min() + 1e-8)
                frame = (frame * 255).astype(np.uint8)
                frame_img = Image.fromarray(frame)

                cam_t = (cam[t] - cam[t].min()) / (cam[t].max() - cam[t].min() + 1e-8)
                cam_t = np.uint8(255 * cam_t)
                cam_color = Image.fromarray(np.uint8(plt.cm.jet(cam_t)[:, :, :3] * 255))

                blended = Image.blend(frame_img.convert("RGBA"), cam_color.convert("RGBA"), alpha=0.5)
                vis_frames.append(blended.convert("RGB"))

            center_num_int = int(center_num.item()) if torch.is_tensor(center_num) else int(center_num)

            sample_dir_name = f"{data_name[0]}_{frame_dir[0]}_{center_num_int}"
            save_base_dir = OUTPUT_DIR / situation_type / test_game / bug_type / result_type / sample_dir_name
            save_base_dir.mkdir(parents=True, exist_ok=True)

            gif_save_path = save_base_dir / "result.gif"
            vis_frames[0].save(
                gif_save_path,
                save_all=True,
                append_images=vis_frames[1:],
                duration=100,
                loop=0,
            )

            frames_save_dir = save_base_dir / "frames"
            frames_save_dir.mkdir(parents=True, exist_ok=True)

            for idx, frame_img in enumerate(vis_frames):
                frame_save_path = frames_save_dir / f"{idx}.jpg"
                frame_img.save(frame_save_path)

            print(f"Saved result to -> {save_base_dir}")

if __name__ == "__main__":
    run_gradcam(
        situation_type="cross-title", 
        test_game="TotK", 
        bug_type="All_label",
    )
