import os
import json
import base64
import time
import pandas as pd
from pathlib import Path
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

def load_glitch_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_temporal_sequence(center_frame_name, glitch_list):
    """41枚のシーケンスを1.34秒の時間軸として解析します．"""
    
    try:
        center_num = int(Path(center_frame_name).stem)
    except ValueError:
        return {"frame": center_frame_name, "error": "Invalid frame name format in CSV."}

    glitch_descriptions = "\n".join([
        f"- Name: {g['name']}, Label: {g['label']}, Description: {g['description']}" 
        for g in glitch_list
    ])

    # プロンプト：1.34秒（30fps想定）に修正
    prompt = f"""
Analyze this sequence of up to 41 frames from a Super Mario 64 speedrun.
The frames range from approximately {center_num - 20}.jpg to {center_num + 20}.jpg．

### Temporal Information:
The total duration of this 41-frame sequence is approximately **1.34 seconds** (roughly 30 fps)．
Use this time scale to estimate Mario's velocity and acceleration across the sequence to distinguish between normal movement and movement-based glitches．

### Glitch List:
{glitch_descriptions}

### Task:
Determine if this image sequence contains any glitch from the provided list. 
Observe the movement, velocity, and state changes across the entire set of frames to identify if a specific glitch occurs at any point in this sequence．

Important Rules:
1. If any glitch is identified within this sequence, provide the corresponding 'label'．
2. If the entire sequence shows normal gameplay, or you are not confident, strictly use "Normal" for the label．

Output the result strictly in JSON format with "frame", "label", and "reason"．
(Note: Use "{center_frame_name}" for the "frame" field to identify this batch)．
"""

    content_list = [{"type": "text", "text": prompt}]

    # 全41フレームを収集（すべて detail="high"）
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
                    "content": "You are an expert in SM64 speedrun glitch detection．You are skilled at calculating object velocity from temporal frame sequences (30fps) to identify movement-based glitches．"
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
        print(f"Error analyzing sequence around {center_frame_name}: {e}")
        return {
            "frame": center_frame_name,
            "label": "Error",
            "reason": str(e)
        }

def main():
    glitches = load_glitch_list(GLITCH_LIST_PATH)
    
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    target_frames = df['frame'].tolist()
    
    results = []
    print(f"Starting analysis with {MODEL_NAME} (Context: 41 frames / 1.34s / 30fps)...")

    for center_frame in target_frames:
        print(f"Processing sequence around: {center_frame}")
        
        result = analyze_temporal_sequence(center_frame, glitches)
        results.append(result)
        
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        time.sleep(3)

    print(f"Analysis complete．Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
