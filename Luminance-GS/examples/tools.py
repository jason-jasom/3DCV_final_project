import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kornia
from torch import cos, sin
import imageio.v2 as imageio
import numpy as np

# Curve Mapping: mapping the input tensor with curve
def LUT_mapping(ts, lut_1d):
    t, t_min, t_max = ts[0], ts[1], ts[2]
    t = torch.clamp(t, 0, 1)
    H, W = t.shape[1], t.shape[2]
    range = lut_1d.shape[-1]-1
    t = (t*range).to(torch.int32)
    N_new = H * W
    id = t.to(torch.long).view(N_new)
    out = lut_1d.reshape(range+1)[id]
    out = out.reshape(H, W).unsqueeze(0)
    return out

def gamma_curve(x, g):
    # Power Curve, Gamma Correction
    y = torch.clamp(x, 1e-3, 1)
    y = y ** g
    return y

def s_curve(x, alpha, beta):
    below_alpha = x <= alpha
    epsilon = 1e-3
    
    # 保護 alpha 避免除零
    alpha_safe = torch.clamp(alpha, epsilon, 1.0 - epsilon)
    
    # 保護 x / alpha 避免數值問題
    ratio_below = torch.clamp(x / alpha_safe, 0, 1 - epsilon)
    s_below_alpha = alpha_safe - alpha_safe * ((1 - ratio_below) ** beta)
    
    # 保護 (x - alpha) / (1 - alpha) 避免數值問題
    ratio_above = torch.clamp((x - alpha_safe) / (1 - alpha_safe), 0, 1 - epsilon)
    s_above_alpha = alpha_safe + (1 - alpha_safe) * (ratio_above ** beta)
    
    return torch.where(below_alpha, s_below_alpha, s_above_alpha)


def value_encode(value, max, min):
    return (value-min)/(max-min)

def value_decode(value, max ,min):
    return value*(max-min)+min

def cal_min_max(normal, normal_sign, bias):
    normal_max = normal[:,0,:,:]*normal_sign[:,0,:,:] + normal[:,1,:,:]*normal_sign[:,1,:,:] + normal[:,2,:,:]*normal_sign[:,2,:,:] + bias
    normal_min = normal[:,0,:,:]*(1-normal_sign[:,0,:,:]) + normal[:,1,:,:]*(1-normal_sign[:,1,:,:]) + normal[:,2,:,:]*(1-normal_sign[:,2,:,:]) + bias
    return normal_max, normal_min

def pixel_project_RGB2HSV(img_rgb):
    img_hsv = kornia.color.rgb_to_hsv(img_rgb)
    img_hsv = img_hsv.permute(0,2,3,1) # (B, H, W, C)
    H, S, V = img_hsv[:, :, :, 0], img_hsv[:, :, :, 1], img_hsv[:, :, :, 2]
    return H,S,V

def pixel_project_HSV2RGB(H,S,V):
    img_hsv = torch.stack([H,S,V],dim=-1).permute(0,3,1,2) #(B,C,H,W)
    img_rgb = kornia.color.hsv_to_rgb(img_hsv)
    return img_rgb

# img: [B, H, W, 3], normal:[B, 3] normal2:[B, 2]
def pixel_project(img, normal, normal2, bias):
    
    img = img.permute(0,2,3,1) # (B, H, W, C)
    R, G, B = img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2]
    
    # 防止除以零：如果 normal[:,0] 接近 0，使用一個小的 epsilon
    normal0_safe = torch.where(torch.abs(normal[:,0]) < 1e-6, 
                                torch.sign(normal[:,0]) * 1e-6, 
                                normal[:,0])
    normal2 = torch.stack([-(normal2[:,0]*normal[:,1] + normal2[:,1]*normal[:,2])/normal0_safe, \
                            normal2[:,0], normal2[:,1]],dim=1)
    normal3 = torch.stack([normal[:,1]*normal2[:,2] - normal[:,2]*normal2[:,1], \
                        normal[:,2]*normal2[:,0] - normal[:,0]*normal2[:,2], \
                        normal[:,0]*normal2[:,1] - normal[:,1]*normal2[:,0]], dim=1)

    normal = F.normalize(normal,dim=1).unsqueeze(-1).unsqueeze(-1)
    normal2 = F.normalize(normal2,dim=1).unsqueeze(-1).unsqueeze(-1)
    normal3 = F.normalize(normal3,dim=1).unsqueeze(-1).unsqueeze(-1)
    
    t1 = R * normal[:,0,:,:] + G * normal[:,1,:,:] + B * normal[:,2,:,:] + bias[:, 0].unsqueeze(-1).unsqueeze(-1)
    t2 = R * normal2[:,0,:,:] + G * normal2[:,1,:,:] + B * normal2[:,2,:,:] + bias[:, 1].unsqueeze(-1).unsqueeze(-1) 
    t3 = R * normal3[:,0,:,:] + G * normal3[:,1,:,:] + B * normal3[:,2,:,:] + bias[:, 2].unsqueeze(-1).unsqueeze(-1) 

    normal_sign = torch.clip(torch.sign(normal), 0, 1)
    normal2_sign = torch.clip(torch.sign(normal2), 0, 1)
    normal3_sign = torch.clip(torch.sign(normal3), 0, 1)
    
    t1_max, t1_min = cal_min_max(normal, normal_sign, bias[:, 0].unsqueeze(-1).unsqueeze(-1))
    
    t2_max, t2_min = cal_min_max(normal2, normal2_sign, bias[:, 1].unsqueeze(-1).unsqueeze(-1))
    
    t3_max, t3_min = cal_min_max(normal3, normal3_sign, bias[:, 2].unsqueeze(-1).unsqueeze(-1))
    
    t1 = value_encode(t1,t1_max,t1_min)
    t2 = value_encode(t2,t2_max,t2_min)
    t3 = value_encode(t3,t3_max,t3_min)
    
    return [t1, normal, t1_max, t1_min], [t2, normal2, t2_max, t2_min], [t3, normal3, t3_max, t3_min], bias
     

# Project back images with learnable affine transformation & bias
def pixel_project_back(t1s, t2s, t3s, bias):
    t1 = value_decode(t1s[0], t1s[2], t1s[3]) - bias[:, 0].unsqueeze(-1).unsqueeze(-1)
    t2 = value_decode(t2s[0], t2s[2], t2s[3]) - bias[:, 1].unsqueeze(-1).unsqueeze(-1)
    t3 = value_decode(t3s[0], t3s[2], t3s[3]) - bias[:, 2].unsqueeze(-1).unsqueeze(-1)
    
    R_new = t1*t1s[1][:,0,:,:] + t2*t2s[1][:,0,:,:] + t3*t3s[1][:,0,:,:]
    G_new = t1*t1s[1][:,1,:,:] + t2*t2s[1][:,1,:,:] + t3*t3s[1][:,1,:,:]
    B_new = t1*t1s[1][:,2,:,:] + t2*t2s[1][:,2,:,:] + t3*t3s[1][:,2,:,:]
    
    img_out = torch.stack([R_new,G_new,B_new],dim=-1).permute(0,3,1,2)
    return img_out

def compute_hsv_loss(colors_low, pixels):
    colors_low_h, colors_low_s, colors_low_v = pixel_project_RGB2HSV(colors_low.permute(0,3,1,2))
    pixels_h, pixels_s, pixels_v = pixel_project_RGB2HSV(pixels.permute(0,3,1,2))
    former_term = colors_low_s * colors_low_v * (cos(colors_low_h) + sin(colors_low_h))
    latter_term = pixels_s * pixels_v * (cos(pixels_h) + sin(pixels_h))
    hsv_loss = F.l1_loss(former_term, latter_term)
    return hsv_loss

def read_image_rgb_mean(image_path):
    try:
        img = imageio.imread(image_path)
        if len(img.shape) == 3:
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            r_mean = np.mean(img[:, :, 0])
            g_mean = np.mean(img[:, :, 1])
            b_mean = np.mean(img[:, :, 2])
            return r_mean, g_mean, b_mean
        return None
    except Exception as e:
        print(f"警告: 無法讀取圖像 {image_path}: {str(e)}")
        return None

def compute_wb_gain(r_mean, g_mean, b_mean, target_rgb=None, use_gray_world=False):
    if use_gray_world or target_rgb is None:
        avg = (r_mean + g_mean + b_mean) / 3.0
        r_gain = avg / r_mean if r_mean > 0.01 else 1.0
        g_gain = avg / g_mean if g_mean > 0.01 else 1.0
        b_gain = avg / b_mean if b_mean > 0.01 else 1.0
    else:
        r_target, g_target, b_target = target_rgb
        r_gain = r_target / r_mean if r_mean > 0.01 else 1.0
        g_gain = g_target / g_mean if g_mean > 0.01 else 1.0
        b_gain = b_target / b_mean if b_mean > 0.01 else 1.0
    
    r_gain = np.clip(r_gain, 0.1, 3.0)
    g_gain = np.clip(g_gain, 0.1, 3.0)
    b_gain = np.clip(b_gain, 0.1, 3.0)
    
    return r_gain, g_gain, b_gain

def compute_wb_gains(image_paths, wb_target_rgb=None, use_gray_world=False):
    gains = []
    
    for image_path in image_paths:
        rgb_mean = read_image_rgb_mean(image_path)
        if rgb_mean is None:
            gains.append([1.0, 1.0, 1.0])
            continue
        
        r_mean, g_mean, b_mean = rgb_mean
        r_gain, g_gain, b_gain = compute_wb_gain(
            r_mean, g_mean, b_mean,
            target_rgb=wb_target_rgb,
            use_gray_world=use_gray_world
        )
        gains.append([r_gain, g_gain, b_gain])
    
    return gains

def create_saturation_reference_map(trainset, parser, saturation_reference_dir, exp_name):
    saturation_map = {}

    if not os.path.exists(saturation_reference_dir):
        print(f"警告: saturation_reference_dir 不存在: {saturation_reference_dir}")
        return saturation_map

    for trainset_idx in range(len(trainset)):
        # 獲取實際的圖像索引
        actual_image_idx = trainset.indices[trainset_idx]
        train_image_path = parser.image_paths[actual_image_idx]
        train_filename = os.path.basename(train_image_path)

        base_name_without_ext = os.path.splitext(train_filename)[0]
        base_name = base_name_without_ext.split('_')[0]

        if exp_name == "low":
            reference_filename = f"{base_name}_test_lcdp.png"
        elif exp_name == "over_exp":
            reference_filename = f"{base_name}_test_msec.png"
        elif exp_name == "variance":
            reference_filename = f"{base_name}_test_lcdp.png"
        else:
            reference_filename = train_filename

        reference_image_path = os.path.join(saturation_reference_dir, reference_filename)
        if os.path.exists(reference_image_path):
            saturation_map[trainset_idx] = reference_image_path
        else:
            print(f"警告: 找不到參考圖片 {reference_image_path}，將跳過此圖片的 saturation 替換")

    print(f"已載入 {len(saturation_map)} 張 saturation 參考圖片")
    return saturation_map