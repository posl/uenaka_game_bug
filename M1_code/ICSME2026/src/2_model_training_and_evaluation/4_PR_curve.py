from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PR_LIST_DIR = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "PR_list"
PR_CURVE_DIR = PROJECT_ROOT / "src" / "2_model_training_and_evaluation" / "PR_curve"
PR_CURVE_DIR.mkdir(parents=True, exist_ok=True)

def pr_curve(situation_type, test_game):
    target_dir = PR_LIST_DIR / situation_type / test_game
    random_dir = PR_LIST_DIR / "random" / test_game
    
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return

    csv_files = list(target_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {target_dir}")
        return

    target_order = ["All", "OoB", "SG", "FG", "IG"]
    
    def get_sort_key(file_path):
        filename = file_path.name
        for i, name in enumerate(target_order):
            if name in filename:
                return i
        return 999

    csv_files.sort(key=get_sort_key)

    plt.figure(figsize=(7, 7))

    def get_color(name):
        if "All" in name: return "#8A2BE2"
        if "OoB" in name: return "#DC143C"
        if "SG" in name:  return "#4169E1"
        if "FG" in name:  return "#FF8C00"
        if "IG" in name:  return "#32CD32"
        return "#696969"
        
    def format_threshold(val):
        if np.isinf(val):
            return "\\infty"
        if np.isnan(val):
            return "N/A"
        if val == 0:
            return "0"
        
        if abs(val) < 0.01:
            return f"{val:.2e}"
        
        return f"{val:.3f}"

    for path in csv_files:
        filename = path.name
        bug_type_raw = path.stem
        label_name = bug_type_raw.replace("_label", "")
        
        color = get_color(label_name)

        df = pd.read_csv(path)
        
        recall = df["recall"].values
        precision = df["precision"].values
        thresholds = df["threshold"].values if "threshold" in df.columns else np.full_like(recall, np.nan)
        
        pr_auc = np.trapz(precision, recall)

        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)

        best_idx = np.argmax(f1_scores)
        best_r = recall[best_idx]
        best_p = precision[best_idx]
        best_th = thresholds[best_idx]
        
        th_str = format_threshold(best_th)

        plt.plot(
            recall,
            precision,
            lw=2,
            label=f"{label_name} (AUC={pr_auc:.3f}, $\\theta^*={th_str}$)",
            color=color,
            linestyle="-",
            alpha=1.0
        )

        if len(recall) > 1:
            plt.plot(
                recall[1], 
                precision[1], 
                marker="o",       
                markersize=10,    
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linestyle="None",
                label="_nolegend_"
            )
        
        plt.plot(
            best_r,
            best_p,
            marker="*",
            markersize=15,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="None",
            label="_nolegend_"
        )

        random_path = random_dir / filename
        if random_path.exists():
            df_rand = pd.read_csv(random_path)
            r_recall = df_rand["recall"].values
            r_precision = df_rand["precision"].values
            r_auc = np.trapz(r_precision, r_recall)

            plt.plot(
                r_recall,
                r_precision,
                lw=1.5,
                label=f"{label_name} (Random) (AUC={r_auc:.3f})",
                color=color,
                linestyle="--",
                alpha=0.6
            )

            if len(r_recall) > 1:
                plt.plot(
                    r_recall[1],
                    r_precision[1],
                    marker="o",
                    markersize=10,
                    markerfacecolor='none', 
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    linestyle="None",
                    label="_nolegend_"
                )

    if situation_type == "within-title":
        plot_title = f"{situation_type}_{test_game}"
    elif situation_type == "cross-title":
        if test_game == "BotW":
            plot_title = f"{situation_type}_SM64_to_{test_game}"
        elif test_game == "TotK":
            plot_title = f"{situation_type}_BotW_to_{test_game}"
        else:
            plot_title = f"{situation_type}_to_{test_game}"
    else:
        plot_title = f"{situation_type}_{test_game}"

    plt.title(plot_title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([-0.03, 1.03])
    plt.ylim([-0.03, 1.03])
    
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True)

    save_filename = f"{situation_type}_{test_game}.pdf"
    save_path = PR_CURVE_DIR / save_filename
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved PR curve (Smart Threshold Format) to: {save_path}")

if __name__ == "__main__":
    test_games = ["SM64", "BotW", "TotK"]
    situation_types = ["within-title", "cross-title"]

    for test_game in test_games:
        for situation_type in situation_types:
            if situation_type == "cross-title" and test_game == "SM64":
                continue
            pr_curve(situation_type, test_game)
