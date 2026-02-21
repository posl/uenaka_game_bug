from pathlib import Path
import pandas as pd
import numpy as np
from prts import ts_precision, ts_recall

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_SPLIT_CSV = DATA_DIR / "2_model_training_and_evaluation" / "data_split.csv"
OUTPUT_FILE = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "output_for_paper.csv"
PR_LIST_DIR = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "PR_list"
PR_LIST_DIR.mkdir(parents=True, exist_ok=True)

def get_range_count(binary_array):
    diff = np.diff(np.concatenate([[0], binary_array, [0]]))
    return np.sum(diff == 1)

def evaluate(situation_type, test_game, bug_type,
             output_file=OUTPUT_FILE, random=False):

    df_purpose = pd.read_csv(DATA_SPLIT_CSV)
    test_names = df_purpose.loc[
        df_purpose[f"{situation_type}_{test_game}"] == "test",
        "Data_name"
    ].tolist()

    file_data_list = []

    for name in test_names:
        for path in (DATASET_DIR / name).glob("frame_label*.csv"):
            print(f"Processing file: {path}")
            df = pd.read_csv(path)
            if not random:
                output_col = f"{situation_type}_{bug_type}_output"
            else:
                output_col = "random_output"
            label_col = f"{bug_type}"
            
            file_data_list.append({
                "output": df[output_col].values,
                "label": df[label_col].values
            })

    def compute_global_metrics(threshold):
        total_recall_score_sum = 0.0
        total_r_count = 0
        total_precision_score_sum = 0.0
        total_p_count = 0

        for data in file_data_list:
            labels = data["label"]
            preds = (data["output"] >= threshold).astype(int)
            
            n_r = get_range_count(labels) 
            n_p = get_range_count(preds) 

            if n_r > 0:
                unique_preds = np.unique(preds)
                if len(unique_preds) < 2:
                    r_v = 1.0 if (1 in unique_preds) else 0.0
                else:
                    r_v = ts_recall(labels, preds, alpha=0.5, cardinality="reciprocal", bias="flat")
                total_recall_score_sum += r_v * n_r
                total_r_count += n_r
            
            if n_p > 0:
                unique_preds = np.unique(preds)
                if len(unique_preds) < 2:
                    if (1 in unique_preds):
                        overlap_size = np.sum(labels & preds)
                        p_v = overlap_size / len(preds)
                    else:
                        p_v = 1.0 
                else:
                    p_v = ts_precision(labels, preds, alpha=0, cardinality="reciprocal", bias="flat")
                total_precision_score_sum += p_v * n_p
                total_p_count += n_p

        global_recall = total_recall_score_sum / total_r_count if total_r_count > 0 else 0.0
        global_precision = total_precision_score_sum / total_p_count if total_p_count > 0 else 1.0
        
        return global_precision, global_recall

    precision_05, recall_05 = compute_global_metrics(0.5)
    f1 = (2 * precision_05 * recall_05 / (precision_05 + recall_05)) if (precision_05 + recall_05) > 0 else 0

    all_outputs = np.concatenate([d["output"] for d in file_data_list])
    thresholds = np.sort(np.unique(all_outputs))[::-1]
    
    if len(thresholds) > 500:
        idx = np.linspace(0, len(thresholds) - 1, 500).astype(int)
        thresholds = thresholds[idx]

    prec_list, rec_list, thresh_list = [], [], []
    for t in thresholds:
        p, r = compute_global_metrics(t)
        prec_list.append(p)
        rec_list.append(r)
        thresh_list.append(t)

    precisions = np.array(prec_list)
    recalls = np.array(rec_list)
    thresholds_arr = np.array(thresh_list)

    idx_sorted = np.argsort(recalls)
    recalls = recalls[idx_sorted]
    precisions = precisions[idx_sorted]
    thresholds_arr = thresholds_arr[idx_sorted]

    unique_recalls = np.unique(recalls)
    new_recalls, new_precisions, new_thresholds = [], [], []

    for r in unique_recalls:
        mask = (recalls == r)
        
        current_precs = precisions[mask]
        current_threshs = thresholds_arr[mask]
        
        best_idx = np.argmax(current_precs)
        
        new_recalls.append(r)
        new_precisions.append(current_precs[best_idx])
        new_thresholds.append(current_threshs[best_idx])
    
    final_recalls = np.array(new_recalls)
    final_precisions = np.array(new_precisions)
    final_thresholds = np.array(new_thresholds)

    if final_recalls[0] == 0:
        final_precisions[0] = 1.0
    else:
        final_recalls = np.insert(final_recalls, 0, 0.0)
        final_precisions = np.insert(final_precisions, 0, 1.0)
        final_thresholds = np.insert(final_thresholds, 0, np.inf)

    pr_auc = np.trapz(final_precisions, final_recalls)

    if random:
        save_dir = PR_LIST_DIR / "random" / test_game
    else:
        save_dir = PR_LIST_DIR / situation_type / test_game

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{bug_type}.csv"
    
    pd.DataFrame({
        "recall": final_recalls,
        "precision": final_precisions,
        "threshold": final_thresholds
    }).to_csv(save_path, index=False)
    
    print(f"PR curve data saved to: {save_path}")

    if not random:
        experiment_name = f"{situation_type}_{bug_type}_{test_game}"
    else:
        experiment_name = f"{situation_type}_{bug_type}_{test_game}_random"
    new_row = {
        "experiment": experiment_name,
        "precision": round(precision_05, 3),
        "recall": round(recall_05, 3),
        "PR-AUC": round(pr_auc, 3),
        "F1": round(f1, 3)
    }

    if not output_file.exists():
        pd.DataFrame([new_row]).to_csv(output_file, index=False)
    else:
        df_out = pd.read_csv(output_file)
        df_out = pd.concat([df_out, pd.DataFrame([new_row])], ignore_index=True)
        df_out.to_csv(output_file, index=False)

    return new_row

if __name__ == "__main__":
    experiments = [
        ("within-title", "SM64", "All_label"),
        ("within-title", "SM64", "OoB_label"),
        ("within-title", "SM64", "SG_label"),
        ("within-title", "BotW", "All_label"),
        ("within-title", "BotW", "OoB_label"),
        ("within-title", "BotW", "SG_label"),
        ("within-title", "BotW", "FG_label"),
        ("within-title", "TotK", "All_label"),
        ("within-title", "TotK", "OoB_label"),
        ("within-title", "TotK", "FG_label"),
        ("within-title", "TotK", "IG_label"),
        ("cross-title", "BotW", "All_label"),
        ("cross-title", "BotW", "OoB_label"),
        ("cross-title", "BotW", "SG_label"),
        ("cross-title", "TotK", "All_label"),
        ("cross-title", "TotK", "OoB_label"),
        ("cross-title", "TotK", "FG_label")
    ]
    for situation_type, test_game, bug_type in experiments:
        evaluate(situation_type, test_game, bug_type)
        if situation_type == "within-title":
            evaluate(situation_type, test_game, bug_type, random=True)
