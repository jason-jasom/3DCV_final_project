import sys
import imageio.v2 as imageio
import numpy as np

def rgb_to_cct(r, g, b):
    """將 RGB 值轉換為色溫 (Correlated Color Temperature)"""
    # 正規化 RGB
    if r + g + b == 0:
        return 0
    
    r_norm = r / (r + g + b)
    g_norm = g / (r + g + b)
    
    # 計算色溫 (使用簡化公式)
    n = (r_norm - 0.3320) / (0.1858 - g_norm)
    cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
    
    return max(1000, min(40000, cct))  # 限制在合理範圍

image = imageio.imread(sys.argv[1])[..., :3] / 255.0
rgb_mean = np.mean(image, axis=(0, 1))
cct = rgb_to_cct(rgb_mean[0], rgb_mean[1], rgb_mean[2])
print(f"R: {rgb_mean[0]:.4f}, G: {rgb_mean[1]:.4f}, B: {rgb_mean[2]:.4f}, CCT: {cct:.1f}K")

