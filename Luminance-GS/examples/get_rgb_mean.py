import sys
import imageio.v2 as imageio
import numpy as np

image = imageio.imread(sys.argv[1])[..., :3] / 255.0
rgb_mean = np.mean(image, axis=(0, 1))
print(f"R: {rgb_mean[0]:.4f}, G: {rgb_mean[1]:.4f}, B: {rgb_mean[2]:.4f}")

