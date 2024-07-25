import requests
import json

def get_game_id(game_name):
    """ゲーム名からゲームIDを取得する"""
    base_url = "https://www.speedrun.com/api/v1/games"
    params = {"name": game_name}
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if 'data' in data and data['data']:
        return data['data'][0]['id']
    return None

def get_category_ids(game_id):
    """ゲームIDからカテゴリIDを取得する"""
    base_url = f"https://www.speedrun.com/api/v1/games/{game_id}/categories"
    response = requests.get(base_url)
    data = response.json()

    if 'data' in data:
        category_ids = [category['id'] for category in data['data']]
        return category_ids
    return []

def get_run_details(game_id, category_id):
    """ゲームIDとカテゴリIDからランの詳細を取得する"""
    base_url = f"https://www.speedrun.com/api/v1/leaderboards/{game_id}/category/{category_id}"
    response = requests.get(base_url)
    data = response.json()

    if 'data' in data and 'runs' in data['data']:
        return data['data']['runs']
    return []

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    game_name = input("ゲーム名を入力してください: ")
    game_id = get_game_id(game_name)

    if not game_id:
        print(f"{game_name}のゲームIDが見つかりませんでした。")
        return

    category_ids = get_category_ids(game_id)
    all_runs = []

    for category_id in category_ids:
        runs = get_run_details(game_id, category_id)
        all_runs.extend(runs)

    save_to_json(all_runs, f"{game_name.replace(' ', '_').lower()}_speedrun_info.json")

if __name__ == "__main__":
    main()
