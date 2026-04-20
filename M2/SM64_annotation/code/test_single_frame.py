import os
import json
import base64
import time
from pathlib import Path
from openai import OpenAI

# --- Configuration ---
# APIキーは環境変数 OPENAI_API_KEY から取得します．
# 実行前に export OPENAI_API_KEY='your-key' を行ってください．
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 使用モデル：2026年現在の最新Vision対応モデル
MODEL_NAME = "gpt-4o"

# Relative Paths (SM64_annotation/code/ からの相対パス)
BASE_DIR = Path(__file__).parent
GLITCH_LIST_PATH = BASE_DIR / "../../SM64_glitch_list/glitch_list.json"
IMAGE_DIR = BASE_DIR / "../0Star/25/test_single_frames/"
OUTPUT_PATH = BASE_DIR / "../0Star/25/test_single_frames/gpt_results.json"

def load_glitch_list(path):
    """19個のバグリストをJSONファイルから読み込みます．"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_image(image_path):
    """画像をBase64形式にエンコードします．"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_frame_gpt(image_path, glitch_list):
    """GPT-4oによる解析を実行し，結果をJSONで返します．"""
    base64_image = encode_image(image_path)
    
    # 19個のバグ情報をテキストとして整形
    glitch_descriptions = "\n".join([
        f"- Name: {g['name']}, Label: {g['label']}, Description: {g['description']}" 
        for g in glitch_list
    ])

    # 判定ルール：バグがない場合は "Normal" を出力させるよう指示を強化
    prompt = f"""
Analyze the attached image and determine if any glitch from the provided list is occurring．

Glitch List:
{glitch_descriptions}

Important Rules:
1. If a glitch from the list is clearly identified, provide the corresponding 'label' (e.g., BLJ_label)．
2. If the image shows normal gameplay, or if you are not confident that a specific glitch is occurring, strictly use "Normal" for the label．

Output the result strictly in JSON format with "frame", "label", and "reason"．
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Super Mario 64 speedrun glitches．Analyze the image and identify glitches precisely based on the provided list．"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high" # 細かなバグ（クリッピング等）を捉えるため高解像度モードを使用
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"Error analyzing {image_path.name}: {e}")
        return {
            "frame": image_path.name,
            "label": "Error",
            "reason": str(e)
        }

def main():
    # 1. バグリストの読み込み
    if not GLITCH_LIST_PATH.exists():
        print(f"Error: Glitch list not found at {GLITCH_LIST_PATH}")
        return
    glitches = load_glitch_list(GLITCH_LIST_PATH)
    
    # 2. 画像ファイルの取得
    if not IMAGE_DIR.exists():
        print(f"Error: Image directory not found at {IMAGE_DIR}")
        return
    image_files = sorted([f for f in IMAGE_DIR.glob("*") if f.suffix.lower() in [".jpg", ".png"]])
    
    results = []
    print(f"Starting analysis of {len(image_files)} frames with {MODEL_NAME}...")

    # 3. 各フレームの解析を実行
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        result = analyze_frame_gpt(img_path, glitches)
        results.append(result)
        
        # 保存をこまめに行う（途中で停止してもデータが残るようにします）
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        # Rate Limitを考慮した待機（必要に応じて調整）
        time.sleep(0.5)

    print(f"Analysis complete．Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
