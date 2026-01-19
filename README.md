# Installation

Clone our project and install ***Python(>=3.9.0)*** and then install ***Pytorch*** (we use 1.13.1) first and then (we use gsplat 1.0):

```
pip install git+https://github.com/nerfstudio-project/gsplat@v1.0.0
```
```
cd examples
```
```
pip install -r requirements.txt
```
And ensure numpy version < 2.0 (i.e. pip install numpy==1.24.3)

# ☀️: Dataset setup:
Our Luminance-GS is evluated on 3 datasets (LOM-lowlight, LOM-overexposure and MipNeRF360-varying).

For **LOM** dataset (lowlight and overexposure), please refer to [Aleth-NeRF](https://github.com/cuiziteng/Aleth-NeRF), download link [here](https://drive.google.com/file/d/1orgKEGApjwCm6G8xaupwHKxMbT2s9IAG/view).

For **MipNeRF360-varying** dataset, please download from [g-drive link (8.47GB)](https://drive.google.com/file/d/1x0EHT5z9ZrA6JV7-y8A8ijQNFCRTjVMW/view?usp=sharing).

***Note***: MipNeRF360-varying is a synthesized dataset based on [MipNeRF360 dataset](https://jonbarron.info/mipnerf360/), featuring 360° views and inconsistent lighting across images, making NVS more challenging.

Then datasets should be set up as (under this folder):

```
-- Luminance-GS
    -- data
        -- LOM_full (For NVS under low-light and overexposure)
            -- bike
            -- buu
            -- chair
            -- shrub
            -- sofa
        -- NeRF_360 (For NVS under vary-exposure), we only provide downscale ratio 8 for efficiency
            -- bicycle
                -- images
                -- images_8
                -- images_8_variance
                -- sparse
                -- ...
            -- bonsai
            -- counter
            -- ... (total 7 scenes)
```

# ☀️: Model Training:

For ease of use, please open `./examples/simple_trainer_ours.py` and modify the settings to enable this function.

## Vanilla

```
cd examples
```

For LOM dataset low-light ("buu" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir (place you save weights & results)
```

For LOM dataset over-exposure ("buu" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name over_exp --result_dir (place you save weights & results)
```

For MipNeRF360-varying dataset varying exposure ("bicycle" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/NeRF_360/bicycle --exp_name variance --data_factor 8 --result_dir (place you save weights & results)
```

## HSV

To 180 lines, set hsv to "yes" can enable hsv function. The cmd part is the same as vanilla version.

## ViT

To 178 lines, set vit to "yes" can enable vit function. The cmd part is the same as vanilla version.

## Target Luminance

To 191~193 lines, set 191 line to "yes" and set the luminance bound at 192, 193 line as you like. The cmd part is the same as vanilla version.

## Saturation Compensation(Predefined initialization)

To 187 lines, set saturation to "yes"

## Saturation Compensation(CoTF initialization)

You should install CoTF and use dataset to generate images first. Then still turn on 187 lines to "yes".
Noticed that LOM low light dataset should use lcdp model and LOM overexposure use msec model to generate images.

For LOM dataset low-light ("buu" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --reference-saturation-dir (place of CoTF images) --result_dir (place you save weights & results)
```

For LOM dataset over-exposure ("buu" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name over_exp --reference-saturation-dir (place of CoTF images) --result_dir (place you save weights & results)
```

## White Balance

To 183 lines, set solve_wb to "yes". The cmd part is the same as vanilla version.

## Target Corrlated Color Temperature

To 183 lines, set solve_wb to "yes"

For LOM dataset low-light ("buu" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --wb-reference (place of your reference image) --result_dir (place you save weights & results)
```

For LOM dataset over-exposure ("buu" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name over_exp --wb-reference (place of your reference image) --result_dir (place you save weights & results)
```

For MipNeRF360-varying dataset varying exposure ("bicycle" scene for example):
```
python simple_trainer_ours.py --data_dir ../data/NeRF_360/bicycle --exp_name variance --data_factor 8 --wb-reference (place of your reference image) --result_dir (place you save weights & results)
```
