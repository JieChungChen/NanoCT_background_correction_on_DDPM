## Nano-CT Background Correction on Diffusion model

使用DDPM去除TXM拍攝影像的背景  

Diffusion model (DDPM) for background correction of TXM images.  
(based on SR3 - Super Resolution with Diffusion Probabilistic Model)

## Overview  

In TXM, each pixel of the detector may respond differently to incoming X-rays due to variations in sensitivity and illumination. These discrepancies can lead to image artifacts, such as shading or uneven brightness, which can obscure important details in the sample being imaged. Traditionally, we'll get a reference image from empty space, and use it to remove the background of the raw sample image. However, the reference image will change by time due to the instablity of the light source, which means the reference image should be take as frequently as possible. Because of the limitation of the TXM equipment, we can't get a reference during the imaging process of a mosaic or a tomography. Therefore, we develop a background correction model based on DDPM to achieve high quality TXM images postprocessing without actual reference image.  

![img](figs/background_correction.png)

### Model Architecture 

A pair of TXM images acquired within a short time interval should have the same background. Based on this assumption, we use a diffusion model to extract common features from the image pair and generate a possible background image.

![img](figs/architecture.png)

## Installation  

This model is implemented on `python 3.11` with `torch 2.3.1+cu118` 
To install the required modules:  
```
pip install -r requirements.txt
```

## Quick Start  

### Inference

Download the pre-trained model [here]().

You can use [demo.ipynb](demo.ipynb) to remove the background from the images in [demo_imgs](demo_imgs) as example.  
or  
```
python inference.py --test_img_dir FOLDER_PATH_OF_YOUR_IMGS
```

### Training

single gpu command
```
python main.py
```

multi-gpu command (with 4 gpus) 
```
torchrun --standalone --nproc_per_node=4 main.py
```

## Experiment record  

<details>
<summary>ddpm_pair_base</summary>
模型結構使用較高的channels數及較低的深度，DDPM的參數則用原論文的設定。而且`uncon_ratio`一定要至少設到0.5以上，不然產生出來的reference會很容易抓到樣本的特徵。
</details>

<details>
<summary>ddpm_pair_v2</summary>
增加了模型深度並砍了channel數，
</details>

## Results

* **DDPM_PAIR_BASE** (testing set)  
  
![img](figs/ddpm_pair_base.png)