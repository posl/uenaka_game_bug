import os
import json
import base64
import time
import pandas as pd
from pathlib import Path
from string import Template
from openai import OpenAI

# --- Configuration ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2026年現在の最新モデルを使用
MODEL_NAME = "gpt-5"

# Relative Paths
BASE_DIR = Path(__file__).parent
GLITCH_LIST_PATH = BASE_DIR / "../../SM64_glitch_list/glitch_list.json"
CSV_PATH = BASE_DIR / "../0Star/25/test_41_frames/test_frames.csv"
IMAGE_DIR = BASE_DIR / "../0Star/25/frames1/"
OUTPUT_PATH = BASE_DIR / "../0Star/25/test_41_frames/gpt_results.json"
PROMPT_TEMPLATE_PATH = BASE_DIR / "../../SM64_annotation/code/test_41_frames_prompt.txt"


def load_glitch_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_template(path):
    """プロンプトテンプレートをファイルから読み込みます．"""
    with open(path, 'r', encoding='utf-8') as f:
        return Template(f.read())


def format_glitch_descriptions(taxonomy_data):
    """ネストされたバグリストを階層がわかるテキスト形式に変換します．

    階層ルール:
      - Base Glitch が variants を持つ場合        → Base Glitch > Variant (> Application)
      - Base Glitch が variants を持たず
        applications を直接持つ場合              → Base Glitch > Application (variant は N/A)
      - いずれも持たない場合                      → Base Glitch のみ (variant・application は N/A)
    """
    lines = []

    def process_base_glitch(glitch):
        lines.append(f"  - Base Glitch: {glitch['name']} (label: {glitch['label']})")
        lines.append(f"    Description: {glitch['description']}")

        if "variants" in glitch:
            for variant in glitch["variants"]:
                lines.append(f"    - Variant: {variant['name']} (label: {variant['label']})")
                lines.append(f"      Description: {variant['description']}")
                if "applications" in variant:
                    for app in variant["applications"]:
                        lines.append(f"      - Application: {app['name']} (label: {app['label']})")
                        lines.append(f"        Description: {app['description']}")

        elif "applications" in glitch:
            # variants を経由せず applications を直接持つケース (e.g. Bob-omb Clip)
            lines.append(f"    (No variants. Applications attach directly to this Base Glitch.)")
            for app in glitch["applications"]:
                lines.append(f"    - Application: {app['name']} (label: {app['label']})")
                lines.append(f"      Description: {app['description']}")

    for category_group in taxonomy_data.get("glitch_taxonomy", []):
        lines.append(f"\n[Category: {category_group['category']}]")
        for glitch in category_group.get("glitches", []):
            process_base_glitch(glitch)

    return "\n".join(lines)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_frame_sequence(center_frame_name, taxonomy_data, prompt_template):
    """中心フレームの前後を含むシーケンスをモデルに送り，グリッチ分類結果を返します．"""

    try:
        center_num = int(Path(center_frame_name).stem)
    except ValueError:
        return {"frame": center_frame_name, "error": "Invalid frame name format in CSV."}

    glitch_descriptions = format_glitch_descriptions(taxonomy_data)

    # テンプレートに変数を埋め込んでプロンプトを生成
    prompt = prompt_template.substitute(
        glitch_descriptions=glitch_descriptions,
        center_frame_name=center_frame_name,
    )

    content_list = [{"type": "text", "text": prompt}]

    for i in range(center_num - 20, center_num + 21):
        img_name = f"{i}.jpg"
        img_path = IMAGE_DIR / img_name
        if img_path.exists():
            b64_img = encode_image(img_path)
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": "high"
                }
            })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert annotator for Super Mario 64 speedrun footage. You classify glitches using a fixed hierarchical taxonomy. Always respond in valid JSON only, with no additional text."
                },
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error analyzing {center_frame_name}: {e}")
        return {"frame": center_frame_name, "label": "Error", "reason": str(e)}


def main():
    taxonomy = load_glitch_list(GLITCH_LIST_PATH)
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)

    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    target_frames = df['frame'].tolist()

    results = []
    print(f"Starting hierarchical analysis with {MODEL_NAME}...")

    for center_frame in target_frames:
        print(f"Processing sequence around: {center_frame}")
        result = analyze_frame_sequence(center_frame, taxonomy, prompt_template)
        results.append(result)

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        time.sleep(3)

    print(f"Analysis complete．Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
