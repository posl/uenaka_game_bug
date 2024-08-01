import json
from datetime import datetime
import re

def parse_time(duration):
    """ISO 8601形式のタイムを秒数に変換する"""
    match = re.match(r'^PT((\d+)H)?((\d+)M)?((\d+)S)?$', duration)
    if match:
        hours = int(match.group(2) or 0)
        minutes = int(match.group(4) or 0)
        seconds = int(match.group(6) or 0)
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Invalid duration format: {duration}")

def load_data(filepath):
    """指定されたパスからJSONファイルをロードする"""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def process_data(data):
    """データをカテゴリごとに整理し、条件を満たすデータのみを抽出する"""
    categories = {}
    for entry in data:
        run = entry['run']
        run_id = run['id']
        category_id = run['category']
        date_str = run['date']
        time_str = run['times']['primary']
        video_link = run['videos']['links'][0]['uri'] if run['videos'] and 'links' in run['videos'] and run['videos']['links'] else None
        
        if date_str is None:
            print(f"Skipping entry due to missing date: {entry}")
            continue
        
        if time_str:
            try:
                time_seconds = parse_time(time_str)
            except ValueError as e:
                print(f"Skipping entry due to error in time parsing: {e}")
                continue
            
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError as e:
                print(f"Skipping entry due to error in date parsing: {e}")
                continue
            
            if category_id not in categories:
                categories[category_id] = {'dates': [], 'times': [], 'records': []}
                
            is_record = True
            for record in categories[category_id]['records']:
                if record['date'] < date and record['time'] < time_seconds:
                    is_record = False
                    break
            
            if is_record:
                categories[category_id]['records'].append({'date': date, 'time': time_seconds, 'video_link': video_link, 'run_id': run_id})
                categories[category_id]['dates'].append(date)
                categories[category_id]['times'].append(time_seconds)
    
    return categories

def write_video_links(categories, category_names):
    """各カテゴリの動画リンクをテキストファイルに書き出す"""
    for category_id, info in categories.items():
        category_name = category_names.get(category_id, 'Unknown_Category')
        filename = f"{category_name}_video_links.txt"
        with open(filename, 'w') as file:
            for record in sorted(info['records'], key=lambda x: x['date']):
                if record['video_link']:
                    file.write(f"{record['date'].strftime('%Y-%m-%d')}: {record['video_link']}\n")
                else:
                    print(f"Run ID {record['run_id']} has no video link.")
                    # file.write(f"{record['date'].strftime('%Y-%m-%d')}: No video link available for Run ID {record['run_id']}\n")

def main():
    filepath = '/Users/uenakayuto/main-research/mycode/getspeedrun/super_mario_64_speedrun_info.json'  # 絶対パスを指定
    data = load_data(filepath)
    
    categories = process_data(data)
    
    # カテゴリ名は仮の名前として使っています。実際には、カテゴリIDに対応する名前を取得する必要があります。
    category_names = {
        "wkpoo02r": "120 Star",
        "7dgrrxk4": "70 Star",
        "n2y55mko": "16 Star",
        "7kjpp4k3": "1 Star",
        "xk9gg6d0": "0 Star",
    }
    
    # 各カテゴリの動画リンクをテキストファイルに書き出す
    write_video_links(categories, category_names)

if __name__ == "__main__":
    main()
