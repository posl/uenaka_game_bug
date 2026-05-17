import os
import json
import base64
import time
import io
import mimetypes
import pandas as pd
from pathlib import Path
from string import Template
from openai import OpenAI
from PIL import Image, ImageSequence

# --- Configuration ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2026年現在の最新モデルを使用
MODEL_NAME = "gpt-5"

# Relative Paths
BASE_DIR = Path(__file__).parent
GLITCH_LIST_PATH = BASE_DIR / "../../SM64_glitch_list/glitch_list.json"
CSV_PATH = BASE_DIR / "../0Star/25/test_41_frames/test_frames.csv"
IMAGE_DIR = BASE_DIR / "../0Star/25/frames1/"
OUTPUT_DIR = BASE_DIR / "../0Star/25/test_41_frames/23_to_25_results/"
PROMPT_TEMPLATE_PATH = BASE_DIR / "../../SM64_annotation/code/test_41_frames_prompt.txt"
FRONT_AND_BACK_FRAME_COUNT = 20

def load_glitch_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt_template(path):
    """プロンプトテンプレートをファイルから読み込みます．"""
    with open(path, 'r', encoding='utf-8') as f:
        return Template(f.read())

def collect_examples_from_taxonomy(taxonomy_data):
    """タクソノミーを再帰的に探索し，example_path があるものを (名前, パス) のリストで返します．"""
    examples = []

    def walk(obj):
        if isinstance(obj, dict):
            if "name" in obj and "example_path" in obj:
                examples.append((obj["name"], obj["example_path"]))
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(taxonomy_data)
    return examples

def format_glitch_descriptions(taxonomy_data):
    """ネストされたバグリストを階層がわかるテキスト形式に変換します．"""
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
            lines.append(f"    (No variants．Applications attach directly to this Base Glitch．)")
            for app in glitch["applications"]:
                lines.append(f"    - Application: {app['name']} (label: {app['label']})")
                lines.append(f"      Description: {app['description']}")

    for category_group in taxonomy_data.get("glitch_taxonomy", []):
        lines.append(f"\n[Category: {category_group['category']}]")
        for glitch in category_group.get("glitches", []):
            process_base_glitch(glitch)

    return "\n".join(lines)

def encode_image(image_path):
    """静止画ファイルをbase64エンコードします．"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_gif_frames(gif_path, max_frames=60):
    """GIFから複数のフレームを抽出し，base64エンコードのリストで返します．"""
    frames_b64 = []
    try:
        with Image.open(gif_path) as img:
            total_frames = getattr(img, 'n_frames', 1)
            indices = [int(i * total_frames / max_frames) for i in range(max_frames)] if total_frames > max_frames else range(total_frames)
            
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                if i in indices:
                    frame = frame.convert("RGB")
                    buffer = io.BytesIO()
                    frame.save(buffer, format="JPEG")
                    frames_b64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    except Exception as e:
        print(f"Error processing GIF {gif_path}: {e}")
    return frames_b64

def analyze_frame_sequence(center_frame_name, taxonomy_data, prompt_template, x=20):
    """中心フレームの前後を含むシーケンスと，タクソノミー内の例示をモデルに送ります．"""
    try:
        center_num = int(Path(center_frame_name).stem)
    except ValueError:
        return {"frame": center_frame_name, "error": "Invalid frame name format．"}

    glitch_descriptions = format_glitch_descriptions(taxonomy_data)
    
    prompt = prompt_template.substitute(
        glitch_descriptions=glitch_descriptions,
        center_frame_name=center_frame_name,
        total_frames=2 * FRONT_AND_BACK_FRAME_COUNT + 1
    )

    content_list = [{"type": "text", "text": prompt}]

    # 1．Few-shot 用の例示画像を追加
    examples = collect_examples_from_taxonomy(taxonomy_data)
    for name, rel_path in examples:
        full_ex_path = GLITCH_LIST_PATH.parent / rel_path
        if full_ex_path.exists():
            content_list.append({"type": "text", "text": f"### Example of {name}:"})
            
            suffix = full_ex_path.suffix.lower()
            if suffix == ".gif":
                gif_frames = encode_gif_frames(full_ex_path, max_frames=60)
                for b64_frame in gif_frames:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}", "detail": "high"}
                    })
            elif suffix in [".jpg", ".jpeg", ".png"]:
                try:
                    # ファイル名が数値であると仮定し，前後 x フレームを取得
                    center_ex_num = int(full_ex_path.stem)
                    ex_dir = full_ex_path.parent
                    
                    for i in range(center_ex_num - x, center_ex_num + x + 1):
                        # 拡張子は元のファイルに合わせる
                        ex_frame_path = ex_dir / f"{i}{suffix}"
                        if ex_frame_path.exists():
                            b64_ex = encode_image(ex_frame_path)
                            content_list.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_ex}",
                                    "detail": "high"
                                }
                            })
                except ValueError:
                    # 数値でない場合は単一画像として処理
                    b64_ex = encode_image(full_ex_path)
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_ex}", "detail": "high"}
                    })
            else:
                # その他の形式（WebP等）
                mime_type, _ = mimetypes.guess_type(full_ex_path)
                b64_ex = encode_image(full_ex_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type or 'image/jpeg'};base64,{b64_ex}", "detail": "high"}
                })

    # 2．テスト対象の41枚のシーケンスを追加
    content_list.append({"type": "text", "text": f"### Test Sequence ({2 * FRONT_AND_BACK_FRAME_COUNT + 1} frames):"})
    for i in range(center_num - FRONT_AND_BACK_FRAME_COUNT, center_num + FRONT_AND_BACK_FRAME_COUNT + 1):
        img_path = IMAGE_DIR / f"{i}.jpg"
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
        debug_html = "<html><body>"
        for item in content_list:
            if item["type"] == "text":
                debug_html += f"<h3>{item['text']}</h3>"
            elif item["type"] == "image_url":
                url = item["image_url"]["url"]
                debug_html += f'<img src="{url}" style="width:100px; border:1px solid red; margin:2px;">'
        debug_html += "</body></html>"

        with open("debug_input.html", "w") as f:
            f.write(debug_html)
        print("Debug HTML saved to debug_input.html．Open this file to check the inputs．")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert annotator for Super Mario 64 speedrun footage．Use the provided visual examples to understand the characteristics of each glitch before analyzing the test sequence．Always respond in valid JSON only．"
                },
                {"role": "user", "content": content_list}
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

    # --- 10回繰り返すループを追加 ---
    for run_idx in range(1, 11):
        # ファイル名を生成（gpt_results_001.json, gpt_results_002.json ...）
        current_output_filename = f"gpt_results_{run_idx:03d}.json"
        current_output_path = OUTPUT_DIR / current_output_filename
        
        results = []
        print(f"\n--- Starting Run {run_idx}/10: {current_output_filename} ---")

        for center_frame in target_frames:
            print(f"Processing sequence around: {center_frame}")
            result = analyze_frame_sequence(center_frame, taxonomy, prompt_template)
            results.append(result)

            # 各実行ごとに進捗を保存
            with open(current_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            # APIのレートリミットを考慮して待機
            time.sleep(3)
        
        print(f"Run {run_idx} complete．Saved to: {current_output_path}")

    print("\nAll 10 runs are complete．")

if __name__ == "__main__":
    main()
