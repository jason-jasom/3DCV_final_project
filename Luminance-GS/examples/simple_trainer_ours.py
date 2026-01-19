import json
import math
import os
import time
from dataclasses import dataclass, field
from turtle import color
from typing import Dict, List, Optional, Tuple
from PIL import Image

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import nerfview

from datasets.traj import generate_interpolated_path
import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    CrossAttention,
    CrossAttention_Curve,
    CrossAttention_VIT,
    CrossAttention_Curve_VIT,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
)

import matplotlib.pyplot as plt
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
gsplat_path = os.path.abspath(os.path.join(current_dir, '..', 'gsplat'))

sys.path.append(gsplat_path)

from rendering_double import rasterization_dual

from tools import (
    pixel_project_RGB2HSV,
    pixel_project_HSV2RGB,
    LUT_mapping,
    pixel_project,
    pixel_project_back,
    compute_hsv_loss,
    compute_wb_gains,
    create_saturation_reference_map,
)
from losses import L_spa, HistogramPriorLoss, gamma_curve, s_curve

from transformers import AutoImageProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the dataset
    data_dir: str = "../data/LOM/bike"
    # data_dir: str = "../data/NeRF_360/bicycle"

    exp_name: str = "low"   # Switch Conditions Here. overexposure: str = "over_exp"; varying exposure: str = "variance"
    
    # Downsample factor for the dataset
    data_factor: int = 1    # data_factor 8 for Mip360 dataset
    # Directory to save results
    result_dir: str = "results_low/ours/bike"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps, max steps 10000 for LOM dataset training
    max_steps: int = 10_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [500, 1500, 3000, 5_000, 7_000, 10_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [1500, 3000, 5_000, 70_000, 10_000])

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    #refine_stop_iter: int = 15_000
    refine_stop_iter: int = 8_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Contrast Level
    constrast_level: float = 0.5

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    use_vit: str = "no"

    use_hsv: str = "yes"

    # solve white balance
    solve_wb: str = "no"
    wb_reference: Optional[str] = None

    # use saturation
    use_saturation_learning: str = "yes"
    saturation_reference_dir: Optional[str] = None
    
    # bounds for histogram prior loss
    use_histogram_bound: str = "no"
    HE_lower_bound: float = 0.0
    HE_upper_bound: float = 1.0
    use_target: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)

cfg = tyro.cli(Config)

if cfg.exp_name in ["low", "over_exp"]:
    from datasets.colmap import Dataset, Parser
else:
    from datasets.colmap_mip360 import Dataset, Parser


def create_splats_with_optimizers(
    points: Tensor,  # [N, 3]
    rgbs: Tensor,  # [N, 3]
    frame_nums: int, # Training Frame Number
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, torch.optim.Optimizer]:
    N = points.shape[0]

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]   point cloud position
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    params = [
        # name, value, lr
        ("means3d", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:    # Color is Here     
        # color is SH coefficients.     
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, 4**2, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)   # rgb to sh
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
        
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    # Eq.3 in our paper, a least-squares formula
    adjust_k = torch.nn.Parameter(torch.ones_like(colors[:, :1, :]), requires_grad=True)    # enhance, for multiply
    adjust_b = torch.nn.Parameter(torch.zeros_like(colors[:, :1, :]), requires_grad=True)   # bias, for add

    params.append(("adjust_k", adjust_k, 2.5e-3))
    params.append(("adjust_b", adjust_b, 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)    # parameter dict

    optimizers = [
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    ]
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.render_dir_depth = f"{cfg.result_dir}/renders_depth"
        os.makedirs(self.render_dir_depth, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            exp_name = cfg.exp_name,
            factor=cfg.data_factor, # down scale ratio
            normalize=True,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(    # Training Set
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val") # Validation Set
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        
        # Model
        feature_dim = 32 if cfg.app_opt else None
        # return GS-parameters & optimizers
        self.splats, self.optimizers = create_splats_with_optimizers(   # basic gaussian splatting
            torch.from_numpy(self.parser.points).float(),
            torch.from_numpy(self.parser.points_rgb / 255.0).float(),
            frame_nums = len(self.trainset),
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            init_opacity=cfg.init_opa,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))
        
        
        self.constrast_level = cfg.constrast_level
        
        # if True:
        #     curve = torch.zeros(255)
        #     slope_len = 204 - 153 + 1
        #     slopes = torch.linspace(0, 1, slope_len)
        #     curve[153:205] = slopes
        #     curve[205:] = 1.0
        #     curve = curve.unsqueeze(0).cuda()   # Luminance Curve
        # else:   
        curve = torch.linspace(0, 1, 255).unsqueeze(0).cuda()   # Luminance Curve
        self.curve = torch.nn.Parameter(curve)
        self.curve_optimizers = [
                torch.optim.Adam(
                    [self.curve],
                    # [self.curve, self.curve_2, self.curve_3],
                    lr=1e-3 * math.sqrt(cfg.batch_size),
                    weight_decay=1e-4,
                )
            ]
        
        # ========== Vit 設置 ==========
        self.use_vit = (cfg.use_vit == "yes")
        if self.use_vit:
            vit_model_id = "google/vit-base-patch16-384"
            self.processor = AutoImageProcessor.from_pretrained(vit_model_id)
            self.curve_adjust = CrossAttention_VIT(vit_model_id=vit_model_id,freeze=False).to(self.device)    # Output the curve bias parameters, L_k_b
            self.curve_adjust_gamma = CrossAttention_Curve_VIT(vit_model_id=vit_model_id,freeze=False).to(self.device)    # Output the curve shape control parameters, Eq.9 
        else:
            self.curve_adjust = CrossAttention().to(self.device)    # Output the curve bias parameters, L_k_b
            self.curve_adjust_gamma = CrossAttention_Curve().to(self.device)    # Output the curve shape control parameters, Eq.9 
        # ========== Vit 設置結束 ==========

        self.adjust_optimizers = [
                torch.optim.Adam(
                    list(self.curve_adjust.parameters()) + list(self.curve_adjust_gamma.parameters()),
                    lr=1e-5 * math.sqrt(cfg.batch_size),
                    weight_decay=1e-5,
                )
            ]
        
        self.pesdo_curve = torch.nn.Parameter(torch.linspace(0, 1, 255).unsqueeze(0).cuda(), requires_grad=False)
        
        # ========== HSV 設置 ==========
        self.use_hsv = (cfg.use_hsv == "yes")
        if not self.use_hsv:
            self.axis1_para = [torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]).cuda()) for _ in range(len(self.trainset))]
            self.axis2_para = [torch.nn.Parameter(torch.tensor([0.0, 0.0]).cuda()) for _ in range(len(self.trainset))]

            self.sat_optimizers = [
                    torch.optim.Adam(
                        self.axis1_para + self.axis2_para,
                        lr=2e-4 * math.sqrt(cfg.batch_size),
                        weight_decay=1e-4,
                    )
                ]
        # ========== HSV 設置結束 ==========

        self.use_histogram_bound = (cfg.use_histogram_bound == "yes")

        # ========== 白平衡設置和初始化 ==========
        self.solve_wb = (cfg.solve_wb == "yes")
        self.wb_target_rgb = None
        self.wb_gains = []
        self.wb_optimizers = []
        
        if self.solve_wb:
            image_paths = [self.parser.image_paths[self.trainset.indices[i]] for i in range(len(self.trainset))]
            
            if cfg.wb_reference is not None:
                try:
                    ref_image = imageio.v2.imread(cfg.wb_reference).astype(np.float32) / 255.0
                    r_mean = np.mean(ref_image[:, :, 0])
                    g_mean = np.mean(ref_image[:, :, 1])
                    b_mean = np.mean(ref_image[:, :, 2])
                    
                    self.wb_target_rgb = torch.tensor([r_mean, g_mean, b_mean], device=self.device)
                    target_rgb = (r_mean, g_mean, b_mean)
                    initial_gains = compute_wb_gains(image_paths=image_paths, wb_target_rgb=target_rgb, use_gray_world=False)
                except Exception as e:
                    print(f"錯誤: 無法讀取參考圖片 {cfg.wb_reference}: {str(e)}，將不使用白平衡功能")
                    self.solve_wb = False
            else:
                initial_gains = compute_wb_gains(image_paths=image_paths, wb_target_rgb=None, use_gray_world=True)
            
            wb_params_to_optimize = []
            for gain in initial_gains:
                param = torch.nn.Parameter(torch.tensor(gain).cuda())
                self.wb_gains.append(param)
                wb_params_to_optimize.append(param)
            
            self.wb_optimizers = [
                torch.optim.Adam(wb_params_to_optimize, lr=1e-3 * math.sqrt(cfg.batch_size), weight_decay=1e-5)
            ]
        # ========== 白平衡設置結束 ==========
        
        # ========== 飽和度設置 ==========
        self.use_saturation_learning = (cfg.use_saturation_learning == "yes")
        self.saturation_reference_map = {}

        if self.use_saturation_learning:
            if cfg.saturation_reference_dir is not None:
                self.k_high = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
                self.insert = []
                self.saturation_reference_map = create_saturation_reference_map(self.trainset, self.parser, cfg.saturation_reference_dir, cfg.exp_name)
                for trainset_idx in range(len(self.trainset)):
                    if trainset_idx in self.saturation_reference_map:
                        # 讀取原始訓練圖片
                        actual_image_idx = self.trainset.indices[trainset_idx]
                        train_image_path = self.parser.image_paths[actual_image_idx]
                        train_image = imageio.v2.imread(train_image_path)
                        train_image_tensor = torch.from_numpy(train_image.astype(np.float32) / 255.0).to(self.device)
                        train_image_tensor = train_image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                        _, s_original, _ = pixel_project_RGB2HSV(train_image_tensor)  # [1, H, W]
                        s_original_mean = torch.mean(s_original).item()
                        
                        # 讀取參考圖片
                        ref_image_path = self.saturation_reference_map[trainset_idx]
                        ref_image = imageio.v2.imread(ref_image_path)
                        ref_image_tensor = torch.from_numpy(ref_image.astype(np.float32) / 255.0).to(self.device)
                        ref_image_tensor = ref_image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                        _, s_ref, _ = pixel_project_RGB2HSV(ref_image_tensor)  # [1, H, W]
                        s_ref_mean = torch.mean(s_ref).item()
                        
                        s_init = s_ref_mean - s_original_mean
                    else:
                        s_init = 0.0

                    self.insert.append(torch.nn.Parameter(torch.tensor(s_init, device=self.device)))
            else:
                self.k_high = torch.nn.Parameter(torch.tensor(0.1, device=self.device))
                self.insert = []
                for i in range(len(self.trainset)):
                    pixels = self.trainset[i]["image"]
                    pixels = pixels / 255.0
                    h,s,v = pixel_project_RGB2HSV(pixels.unsqueeze(0).permute(0,3,1,2))
                    v_mean = torch.mean(v).item()
                    s_init = 0.5 * v_mean ** 10
                    self.insert.append(torch.nn.Parameter(torch.tensor(s_init, device=self.device)))
                    
            self.saturation_optimizers = [
                torch.optim.Adam(
                    [self.k_high] + self.insert,
                    lr=2e-4 * math.sqrt(cfg.batch_size),
                    weight_decay=1e-4,
                )
            ]
        # ========== 飽和度設置結束 ==========

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Learnable Parameters:
        means = self.splats["means3d"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,], sigmoid function

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:    
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
            
        else:  # Here 
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        adjust_k = self.splats["adjust_k"]  # 1090, 1, 3
        adjust_b = self.splats["adjust_b"]  # 1090, 1, 3
        
        colors_low = colors * adjust_k + adjust_b  # least squares: x_enh=a*x+b
        
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        
        render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info = rasterization_dual(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,    
            colors=colors,
            colors_low=colors_low,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,)
        
        return render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info   # return colors and alphas

    def train(self):
        cfg = self.cfg
        device = self.device

        loss_contrast = L_spa()     # spatial consistancy loss
        loss_histo = HistogramPriorLoss(self.cfg.HE_lower_bound, self.cfg.HE_upper_bound, self.cfg.use_target)    # curve control loss


        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        scheulers = [
            # means3d has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        # curve optimizer & curve adjustment optimizer & sat optimizer
        scheulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.curve_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        scheulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.adjust_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if not self.use_hsv:
            scheulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.sat_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                    )
                )
        if self.use_saturation_learning:
            scheulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.saturation_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                    )
                )
        if self.solve_wb and len(self.wb_optimizers) > 0:
            scheulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.wb_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                    )
                )

        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            scheulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]

            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders_enh, renders_low, alphas_enh, alphas_low, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB",
            )
            if renders_low.shape[-1] == 4:
                colors_low, depths_low = renders_low[..., 0:3], renders_low[..., 3:4]
                colors_enh, depths_enh = renders_enh[..., 0:3], renders_enh[..., 3:4]
            else:
                colors_low, depths_low = renders_low, None
                colors_enh, depths_enh = renders_enh, None

            if cfg.random_bkgd: # False
                bkgd = torch.rand(1, 3, device=device)
                colors_low = colors_low + bkgd * (1.0 - alphas_low)
                colors_enh = colors_enh + bkgd * (1.0 - alphas_enh)

            info["means2d"].retain_grad()  # used for running stats

            ###vit
            if self.use_vit:
                img_PIL = Image.fromarray(data["image"][0].numpy().astype(np.uint8))
                vit_inputs = self.processor(img_PIL, return_tensors="pt").to(device)
                vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}
                curve_adj_bias = self.curve_adjust(vit_inputs, camtoworlds) # encode low-light GT and camera position to get adjust curve
                gamma_alpha_beta = self.curve_adjust_gamma(vit_inputs, camtoworlds)
            else:
                curve_adj_bias = self.curve_adjust(pixels.permute(0,3,1,2), camtoworlds) # encode low-light GT and camera position to get adjust curve
                gamma_alpha_beta = self.curve_adjust_gamma(pixels.permute(0,3,1,2), camtoworlds)
            ###


            curve_adj = torch.clamp(self.curve + curve_adj_bias, 0, 1)    # Clamp the curve in range of (0, 1)

            ###hsv
            if self.use_hsv:
                h,s,v = pixel_project_RGB2HSV(pixels.permute(0,3,1,2))
                v_out = LUT_mapping([v,0,0], curve_adj)
                if self.use_saturation_learning:
                    v_mean = torch.mean(v).item()
                    s_gain = 1.0 + self.k_high * (v_mean ** 10)
                    if v_mean > 0.75:
                        s_insert = self.insert[image_ids]
                    else:
                        s_insert = 0.0
                    s_out = s*s_gain + s_insert
                    s_out = torch.clamp(s_out, 0.0, 1.0)
                    s = s_out
                    if step % 100 == 0:
                        print(f"k_high: {self.k_high.item()}, insert: {self.insert[image_ids].item()}")
                pixels_enh = pixel_project_HSV2RGB(h,s,v_out).permute(0,2,3,1)
            else:
                normal= (self.axis1_para[image_ids] + torch.Tensor([1, 0, 0]).to(colors_low.device)).unsqueeze(0)
                normal2 = (self.axis2_para[image_ids] + torch.Tensor([1, 0]).to(colors_low.device)).unsqueeze(0)
                bias = torch.zeros([1, 3]).to(colors_low.device)

                t1s, t2s, t3s, bias  = pixel_project(pixels.permute(0,3,1,2), normal, normal2, bias)
                t1s_out = [LUT_mapping(t1s, curve_adj), t1s[1], t1s[2], t1s[3]] 
                t2s_out = [LUT_mapping(t2s, curve_adj), t2s[1], t2s[2], t2s[3]] 
                t3s_out = [LUT_mapping(t3s, curve_adj), t3s[1], t3s[2], t3s[3]] 

                pixels_enh = pixel_project_back(t1s_out, t2s_out, t3s_out, bias).permute(0,2,3,1)
            ###
            
            ### White Balance
            if self.solve_wb:
                batch_size = pixels_enh.shape[0]
                wb_gains_list = [self.wb_gains[image_ids[i].item()] for i in range(batch_size)]
                wb_gain = torch.stack(wb_gains_list, dim=0)
                wb_gain = torch.clamp(wb_gain, min=0.1, max=3.0)
                pixels_enh = pixels_enh * wb_gain.unsqueeze(1).unsqueeze(1)
                pixels_enh = torch.clamp(pixels_enh, 0.0, 1.0)
            ###

            def adjust_luminance_to_target(img, target_lum=0.5):
                """
                img: [1, H, W, 3], RGB, float32, range [0–1], *still in sRGB gamma space*
                """

                def tone_aces(x):
                    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
                    x = np.maximum(x, 0.0) 
                    y = (x*(a*x+b)) / (x*(c*x+d)+e)
                    return np.clip(y, 0.0, 1.0)
                
                def mean_after_scale(scale):
                    V_scaled = lum * scale
                    V_mapped = tone_aces(V_scaled)
                    return float(V_mapped.mean())
                
                img_rgb = img[0]
                
                linear = img_rgb

                lum = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
                
                
                current_lum = np.mean(lum)

                scale = target_lum / current_lum
                
                lo, hi = 0.01, 200.0
                for _ in range(30):
                    mid = (lo + hi) / 2
                    mv = mean_after_scale(mid)
                    if mv < target_lum:
                        lo = mid
                    else:
                        hi = mid

                scale = (lo + hi) / 2

                # Scale entire linear image
                linear_scaled = linear * scale
                
                linear_scaled = tone_aces(linear_scaled)
                
                final_lum = 0.2126 * linear_scaled[..., 0] + 0.7152 * linear_scaled[..., 1] + 0.0722 * linear_scaled[..., 2]
                print("Final LUM", np.mean(final_lum))

                srgb_scaled = linear_scaled

                srgb_scaled = np.clip(srgb_scaled, 0.0, 1.0)

                # Add batch dimension back
                return srgb_scaled[np.newaxis, ...]
            
            if self.use_histogram_bound:
                pixels_enh = adjust_luminance_to_target(pixels_enh.detach().cpu().numpy(), ((self.cfg.HE_lower_bound + self.cfg.HE_upper_bound)/2))
                pixels_enh = torch.from_numpy(pixels_enh).float().to(device)
            
            gamma = gamma_alpha_beta[:,0]
            alpha, beta = gamma_alpha_beta[:,1], gamma_alpha_beta[:,2]
            
            gamma = torch.Tensor([1.0]).to(device) + 0.1*gamma
            alpha = torch.Tensor([0.5]).to(device) + 0.002*alpha 
            beta = torch.Tensor([1.0]).to(device) + 0.002*beta
            
            pesdo_curve = gamma_curve(self.pesdo_curve, gamma)  # Pseudo-gamma curve
            pesdo_curve = s_curve(pesdo_curve, alpha, beta) # Pseudo-scurve curve
            

            con_degree = (self.constrast_level/torch.mean(pixels)).item()   # frame-adaptive contrast degree, Eq.8 in paper
            loss_co = loss_contrast(pixels.permute(0,3,1,2), colors_enh.permute(0,3,1,2), contrast=con_degree)
            
            l1loss = F.l1_loss(colors_low, pixels)
            ssimloss = 1.0 - self.ssim(pixels.permute(0,3,1,2), colors_low.permute(0,3,1,2))
            loss_regress_low = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            l1loss_enh = F.l1_loss(colors_enh, pixels_enh)  # enhancement loss constrain
            ssimloss_enh = 1.0 - self.ssim(pixels_enh.permute(0,3,1,2), colors_enh.permute(0,3,1,2))
            loss_regress_enh = l1loss_enh * (1.0 - cfg.ssim_lambda) + ssimloss_enh * cfg.ssim_lambda
            
            if self.use_histogram_bound:
                if self.use_hsv:
                    hist_loss = loss_histo(curve_adj, pixels, pesdo_curve, step, type="target_hsv")
                else:
                    hist_loss = loss_histo(curve_adj, pixels, pesdo_curve, step, type="target_rgb")
            else:
                if self.use_hsv:
                    hist_loss = loss_histo(curve_adj, pixels, pesdo_curve, step, type="hsv")
                else:
                    hist_loss = loss_histo(curve_adj, pixels, pesdo_curve, step, type="rgb")
            
            loss_wb = 0.0
            if self.solve_wb:
                r_mean = torch.mean(pixels_enh[..., 0]) 
                g_mean = torch.mean(pixels_enh[..., 1])
                b_mean = torch.mean(pixels_enh[..., 2])
                
                if self.wb_target_rgb is not None:
                    target_r, target_g, target_b = self.wb_target_rgb[0], self.wb_target_rgb[1], self.wb_target_rgb[2]
                    target_rb_ratio, target_bg_ratio, target_gr_ratio = target_r / target_b, target_b / target_g, target_g / target_r
                    rb_ratio, bg_ratio, gr_ratio = r_mean / b_mean, b_mean / g_mean, g_mean / r_mean
                    loss_wb = torch.clamp(abs(rb_ratio - target_rb_ratio), 0.0, 1.0) + torch.clamp(abs(bg_ratio - target_bg_ratio), 0.0, 1.0) + torch.clamp(abs(gr_ratio - target_gr_ratio), 0.0, 1.0)
                else:
                    loss_wb = abs(r_mean - g_mean) + abs(g_mean - b_mean) + abs(b_mean - r_mean)
                
            loss_saturation = 0.0
            if self.use_saturation_learning:
                if self.k_high < 0 or self.k_high > 0.2:
                    loss_saturation += abs(self.k_high)
                if self.insert[image_ids] < -0.1 or self.insert[image_ids] > 0.3:
                    loss_saturation += abs(self.insert[image_ids])

            loss = loss_regress_low + 0.5*loss_regress_enh + loss_co + 10 * hist_loss + 10 * loss_saturation + 10 * loss_wb
            
            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means3d"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                
                if cfg.tb_save_image:
                    canvas = torch.cat([colors_enh, pixels_enh], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)

                    canvas_low = torch.cat([colors_low, pixels], dim=2).detach().cpu().numpy()
                    canvas_low = canvas_low.reshape(-1, *canvas_low.shape[2:])
                    self.writer.add_image("train/render_low", canvas_low, step)

                self.writer.flush()

            # update running stats for prunning & growing
            if step < cfg.refine_stop_iter:
                self.update_running_stats(info)

                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    # grow GSs
                    is_grad_high = grads >= cfg.grow_grad2d
                    is_small = (
                        torch.exp(self.splats["scales"]).max(dim=-1).values
                        <= cfg.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < cfg.prune_opa
                    if step > cfg.reset_every:
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # reset running stats
                    self.running_stats["grad2d"].zero_()
                    self.running_stats["count"].zero_()

                if step % cfg.reset_every == 0:
                    self.reset_opa(cfg.prune_opa * 2.0)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.curve_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.adjust_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if not self.use_hsv:
                for optimizer in self.sat_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if self.use_saturation_learning:
                for optimizer in self.saturation_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if self.solve_wb:
                for optimizer in self.wb_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in scheulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step, pixels_enh)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def update_running_stats(self, info: Dict):
        """Update running stats."""
        cfg = self.cfg

        # normalize grads to [-1, 1] screen space
        if cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size
        if cfg.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids))
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self, step: int, pixels_enh):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "luminance":[]}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            torch.cuda.synchronize()
            tic = time.time()
            colors_enh, colors_low, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            
            depth_low = colors_low[:, :, :, 3:]
            depth_enh = colors_enh[:, :, :, 3:]
            colors_low = colors_low[:, :, :, :3]
            
            colors_enh = torch.clamp(colors_enh[:, :, :, :3], 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            
            canvas = torch.cat([colors_low, colors_enh], dim=2).squeeze(0).cpu().numpy()
            
            imageio.imwrite(
                f"{self.render_dir_depth}/val_{i:04d}_depth_low.png", (depth_low.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            )

            imageio.imwrite(
                f"{self.render_dir_depth}/val_{i:04d}_depth_enh.png", (depth_enh.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_low.png", (colors_low.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_enh.png", (colors_enh.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            )
                        
                        
            R, G, B = colors_enh[..., 0], colors_enh[..., 1], colors_enh[..., 2]
            
            luma_tensor = 0.2126 * R + 0.7152 * G + 0.0722 * B
            
            metrics["luminance"].append(torch.mean(luma_tensor).flatten().item())

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_enh = colors_enh.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors_enh, pixels))
            metrics["ssim"].append(self.ssim(colors_enh, pixels))
            metrics["lpips"].append(self.lpips(colors_enh, pixels))
            # metrics["luminance"].append(torch.mean(torch.mean(colors_enh, dim=1).flatten()).item())

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        luminance = sum(metrics["luminance"])/len(metrics["luminance"])
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
            f" Average Luminance: {luminance:.3f}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
            "luminance": torch.mean(torch.mean(colors_enh, dim=1).flatten()).item(),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        # print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        camtoworlds = self.parser.camtoworlds[10:60]
        camtoworlds = generate_interpolated_path(camtoworlds, 10)  # [N, 3, 4]
        
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]
        
        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders_enh, renders_low, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders_enh[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders_enh[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            canvas = colors
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=50)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors_enh, render_colors_low, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors_enh[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
