import json
from datetime import datetime
import re

def load_data(filepath):
    """指定されたパスからJSONファイルをロードする"""
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filepath}")
        return None


def parse_duration(duration_str):
    """ISO 8601 Duration (e.g., "PT30M45.5S") を秒に変換"""
    time_units = {'H': 3600, 'M': 60, 'S': 1}
    total_seconds = 0
    matches = re.findall(r'(\d+\.?\d*)([HMS])', duration_str)
    for value, unit in matches:
        total_seconds += float(value) * time_units[unit]
    return total_seconds

def process_data(data):
    """カテゴリごとにランのデータを整理"""
    categories = {}
    
    for run_data in data:
        run = run_data.get('run', {})
        category = run.get('category')
        date_str = run.get('date')
        run_time = run.get('times', {}).get('primary')
        videos = run.get('videos', {})
        video_link = None
        
        if videos and 'links' in videos and videos['links']:
            video_link = videos['links'][0].get('uri')
        
        run_id = run.get('id')

        # チェックを追加して、必要なフィールドが存在するか確認
        if date_str is None or run_time is None:
            print(f"Missing date or time in run data: {run_id}")
            continue
        
        try:
            run_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            print(f"Date format error for run ID {run_id}: {date_str} ({e})")
            continue
        
        run_time_seconds = parse_duration(run_time)
        
        if category not in categories:
            categories[category] = []

        categories[category].append({
            'date': run_date,
            'time': run_time_seconds,
            'video_link': video_link,
            'run_id': run_id
        })
    
    for category in categories:
        categories[category].sort(key=lambda x: x['date'])

    # 各カテゴリ内で重複を削除
    results = {}
    for category, runs in categories.items():
        results[category] = []
        for run in runs:
            is_new_record = True
            for existing_run in results[category]:
                if existing_run['time'] <= run['time']:
                    is_new_record = False
                    break
            if is_new_record:
                results[category].append(run)
    
    return results

def write_video_links(categories, category_names):
    """各カテゴリの動画リンクをテキストファイルに書き出す"""
    for category_id, records in categories.items():
        category_name = category_names.get(category_id, 'Unknown_Category')
        filename = f"{category_name}_video_links_kai.txt"
        with open(filename, 'w') as file:
            for record in sorted(records, key=lambda x: x['date']):
                if record['video_link']:
                    file.write(f"{record['date'].strftime('%Y-%m-%d')}: {record['video_link']}\n")
                else:
                    print(f"Run ID {record['run_id']} has no video link.")

def main():
    filepath = '/Users/uenakayuto/main-research/mycode/getspeedrun/super_mario_64_speedrun_info.json'
    data = load_data(filepath)
    if data is None:
        return
    
    categories = process_data(data)
    
    category_names = {
        "wkpoo02r": "120 Star",
        "7dgrrxk4": "70 Star",
        "n2y55mko": "16 Star",
        "7kjpp4k3": "1 Star",
        "xk9gg6d0": "0 Star",
    }
    
    write_video_links(categories, category_names)

if __name__ == "__main__":
    main()
