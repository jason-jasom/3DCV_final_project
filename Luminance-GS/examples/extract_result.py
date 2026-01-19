import json
import os
from pathlib import Path

# 設定結果目錄路徑
results_dir = Path("../results")
output_file = Path("results.md")

# 需要處理的資料夾類型
folders = ["low", "high", "variance"]
type_mapping = {"low": "low", "high": "high", "variance": "var"}

# 儲存結果
results = []

# 遍歷每個資料夾類型
for folder_type in folders:
    folder_path = results_dir / folder_type
    
    if not folder_path.exists():
        continue
    
    # 取得該類型下的所有資料集資料夾
    dataset_folders = [d for d in folder_path.iterdir() if d.is_dir()]
    
    for dataset_folder in dataset_folders:
        dataset_name = dataset_folder.name
        
        # 讀取 stats/val_step9999.json
        json_file = dataset_folder / "stats" / "val_step9999.json"
        
        if not json_file.exists():
            print(f"警告: {json_file} 不存在，跳過")
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 提取前三行資料（psnr, ssim, lpips）
            psnr = round(data.get("psnr", 0), 3)
            ssim = round(data.get("ssim", 0), 4)
            lpips = round(data.get("lpips", 0), 3)
            
            # 格式化資料集名稱（首字母大寫）
            formatted_name = dataset_name.capitalize()
            
            # 取得類型標籤
            type_label = type_mapping[folder_type]
            
            # 儲存結果
            results.append({
                "dataset": formatted_name,
                "type": type_label,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips
            })
            
        except Exception as e:
            print(f"錯誤: 讀取 {json_file} 時發生問題: {e}")

# 定義排序順序
type_order = {"low": 0, "high": 1, "var": 2}

dataset_order_low = ["Bike", "Buu", "Chair", "Shrub", "Sofa"]
dataset_order_high = ["Bike", "Buu", "Chair", "Shrub", "Sofa"]
dataset_order_var = ["Bicycle", "Bonsai", "Counter", "Garden", "Kitchen", "Room", "Stump"]

def get_sort_key(result):
    type_rank = type_order.get(result["type"], 999)
    
    # 根據類型選擇對應的資料集順序
    if result["type"] == "low":
        dataset_rank = dataset_order_low.index(result["dataset"]) if result["dataset"] in dataset_order_low else 999
    elif result["type"] == "high":
        dataset_rank = dataset_order_high.index(result["dataset"]) if result["dataset"] in dataset_order_high else 999
    elif result["type"] == "var":
        dataset_rank = dataset_order_var.index(result["dataset"]) if result["dataset"] in dataset_order_var else 999
    else:
        dataset_rank = 999
    
    return (type_rank, dataset_rank)

# 依指定順序排序
results.sort(key=get_sort_key)

# 寫入 results.md
with open(output_file, 'w', encoding='utf-8') as f:
    current_type = None
    for result in results:
        # 如果類型改變，且不是第一個，添加空行
        if current_type is not None and current_type != result["type"]:
            f.write("\n")
        current_type = result["type"]
        
        line = f"{result['dataset']} {result['type']} {result['psnr']:.3f} {result['ssim']:.4f} {result['lpips']:.3f}\n"
        f.write(line)

print(f"已成功將結果寫入 {output_file}")
print(f"共處理 {len(results)} 個資料集")

