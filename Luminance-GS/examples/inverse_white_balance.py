#!/usr/bin/env python3
"""
白平衡處理工具
對輸入資料夾中的所有 .jpg/.JPG 圖像應用隨機的 R, G, B gain
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import random


def apply_white_balance(image, r_gain, g_gain, b_gain):
    """
    對圖像應用白平衡增益
    
    Args:
        image: PIL Image 對象
        r_gain: 紅色通道增益
        g_gain: 綠色通道增益
        b_gain: 藍色通道增益
    
    Returns:
        處理後的 PIL Image 對象
    """
    # 轉換為 numpy 數組
    img_array = np.array(image, dtype=np.float32)
    
    # 確保圖像是 RGB 模式
    if len(img_array.shape) == 2:  # 灰度圖
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # 應用增益
    img_array[:, :, 0] *= r_gain  # R
    img_array[:, :, 1] *= g_gain  # G
    img_array[:, :, 2] *= b_gain  # B
    
    # 限制值在 0-255 範圍內
    img_array = np.clip(img_array, 0, 255)
    
    # 轉換回 uint8
    img_array = img_array.astype(np.uint8)
    
    # 轉換回 PIL Image
    return Image.fromarray(img_array)


def process_images(input_dir, output_dir, r_range=(1.0, 1.0), g_range=None, b_range=None, seed=None):
    """
    處理資料夾中的所有 .jpg/.JPG 圖像
    
    Args:
        input_dir: 輸入資料夾路徑
        output_dir: 輸出資料夾路徑
        r_range: 紅色通道增益範圍 (min, max)，預設為 (1.0, 1.0)
        g_range: 綠色通道增益範圍 (min, max)，如果為 None 則使用預設值 (1.0, 1.0)
        b_range: 藍色通道增益範圍 (min, max)，如果為 None 則使用預設值 (1.0, 1.0)
        seed: 隨機種子（可選，用於可重現性）
    """
    # 設置隨機種子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 如果 g_range 或 b_range 未指定，使用預設值 (1.0, 1.0)，不改變顏色
    if g_range is None:
        g_range = (1.0, 1.0)
    if b_range is None:
        b_range = (1.0, 1.0)
    
    # 轉換為 Path 對象
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 檢查輸入資料夾是否存在
    if not input_path.exists():
        raise ValueError(f"輸入資料夾不存在: {input_dir}")
    
    # 創建輸出資料夾
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 .jpg 和 .JPG 文件
    image_extensions = ['.jpg', '.JPG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
    
    if not image_files:
        print(f"警告: 在 {input_dir} 中沒有找到 .jpg 或 .JPG 文件")
        return
    
    print(f"找到 {len(image_files)} 個圖像文件")
    print(f"R 增益範圍: {r_range[0]:.2f} - {r_range[1]:.2f}")
    print(f"G 增益範圍: {g_range[0]:.2f} - {g_range[1]:.2f}")
    print(f"B 增益範圍: {b_range[0]:.2f} - {b_range[1]:.2f}")
    
    # 處理每個圖像
    for i, img_path in enumerate(image_files, 1):
        try:
            # 讀取圖像
            image = Image.open(img_path)
            
            # 生成隨機增益（每個通道使用各自的範圍）
            r_gain = random.uniform(r_range[0], r_range[1])
            g_gain = random.uniform(g_range[0], g_range[1])
            b_gain = random.uniform(b_range[0], b_range[1])
            
            # 應用白平衡
            processed_image = apply_white_balance(image, r_gain, g_gain, b_gain)
            
            # 保存處理後的圖像
            output_file = output_path / img_path.name
            processed_image.save(output_file, quality=95)
            
            print(f"[{i}/{len(image_files)}] {img_path.name} - "
                  f"R:{r_gain:.3f}, G:{g_gain:.3f}, B:{b_gain:.3f}")
            
        except Exception as e:
            print(f"錯誤: 處理 {img_path.name} 時發生問題: {str(e)}")
            continue
    
    print(f"\n處理完成！輸出資料夾: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='對圖像應用隨機白平衡增益',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 使用預設範圍（所有通道 1.0-1.0，不改變圖像）
  python white.py --input ./images --output ./output
  
  # 所有通道使用相同範圍
  python white.py --input ./images --output ./output --gain_min 0.9 --gain_max 1.1
  
  # 分別指定 R, G, B 範圍
  python white.py --input ./images --output ./output --r_range 0.8 1.2 --g_range 0.9 1.1 --b_range 0.85 1.15
  
  # 混合使用（R 和 G 分別指定，B 使用預設）
  python white.py --input ./images --output ./output --r_range 0.7 1.3 --g_range 0.9 1.1
  
  # 使用隨機種子
  python white.py --input ./images --output ./output --r_range 0.8 1.2 --seed 42
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='輸入資料夾路徑'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='輸出資料夾路徑'
    )
    
    parser.add_argument(
        '--gain_min',
        type=float,
        default=1.0,
        help='最小增益值（所有通道，預設: 1.0）'
    )
    
    parser.add_argument(
        '--gain_max',
        type=float,
        default=1.0,
        help='最大增益值（所有通道，預設: 1.0）'
    )
    
    parser.add_argument(
        '--gain_range',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='增益範圍（所有通道，會覆蓋 --gain_min 和 --gain_max）'
    )
    
    parser.add_argument(
        '--r_range',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='紅色通道增益範圍 (會覆蓋 --gain_min/--gain_max 或 --gain_range)'
    )
    
    parser.add_argument(
        '--g_range',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='綠色通道增益範圍 (如果未指定則使用預設值 1.0-1.0，不改變顏色)'
    )
    
    parser.add_argument(
        '--b_range',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='藍色通道增益範圍 (如果未指定則使用預設值 1.0-1.0，不改變顏色)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='隨機種子（用於可重現性）'
    )
    
    args = parser.parse_args()
    
    # 確定基礎增益範圍（用於 R 通道，如果未單獨指定）
    if args.r_range:
        r_range = tuple(args.r_range)
    elif args.gain_range:
        r_range = tuple(args.gain_range)
    else:
        r_range = (args.gain_min, args.gain_max)
    
    # 確定 G 和 B 通道的範圍
    g_range = tuple(args.g_range) if args.g_range else None
    b_range = tuple(args.b_range) if args.b_range else None
    
    # 驗證所有增益範圍
    ranges_to_check = [('R', r_range)]
    if g_range:
        ranges_to_check.append(('G', g_range))
    if b_range:
        ranges_to_check.append(('B', b_range))
    
    for channel_name, gain_range in ranges_to_check:
        if gain_range[0] > gain_range[1]:
            raise ValueError(f"{channel_name} 通道：最小增益值不能大於最大增益值")
        if gain_range[0] < 0:
            raise ValueError(f"{channel_name} 通道：增益值不能為負數")
    
    # 處理圖像
    process_images(
        input_dir=args.input,
        output_dir=args.output,
        r_range=r_range,
        g_range=g_range,
        b_range=b_range,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

