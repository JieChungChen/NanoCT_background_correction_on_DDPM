import torch
import torch.nn.functional as F
import os, glob, random, yaml, argparse
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["ytick.direction"] = 'in'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}' 
params = {
   'axes.labelsize': 10,
   'font.size': 12,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'mathtext.fontset': "stix",
   'font.family': "Times New Roman"
   }
plt.rcParams.update(params)
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from data_preprocess import NanoCT_Pair_Dataset
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import DDIM_Sampler
from utils import min_max_norm, calc_psnr, mosaic


def get_args_parser():
    parser = argparse.ArgumentParser('diffusion for background correction', add_help=False)
    parser.add_argument('--configs', default='configs/ddpm_pair_v3.yml', type=str)
    parser.add_argument('--model_ckpt', default='checkpoints/ddpm_pair_v3_400K.pt', type=str)
    parser.add_argument('--test_img_dir', default='test_data/mam_amber_m02.tif', type=str)
    parser.add_argument('--img_save_dir', default='figs_temp', type=str)
    return parser


def plot_2x3(input_imgs, obj_pred_1, obj_pred_2, obj_true_1, obj_true_2, ref_pred, i, save_dir):
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(8, 4))
    gs = axs[0, -1].get_gridspec()
    for ax in axs[:, -1]:
        ax.remove()

    axs[0, 0].set_title(r'$\boldsymbol{I_1}$')
    axs[0, 0].axis('off')
    axs[0, 0].imshow(input_imgs[0], cmap='gray')

    axs[1, 0].set_title(r'$\boldsymbol{I_2}$')
    axs[1, 0].axis('off')
    axs[1, 0].imshow(input_imgs[1], cmap='gray')

    axs[0, 1].set_title(r'$\boldsymbol{S_1}$')
    axs[0, 1].axis('off')
    obj_true_1 = min_max_norm(obj_true_1)
    axs[0, 1].imshow(obj_true_1, cmap='gray')

    axs[1, 1].set_title(r'$\boldsymbol{S_2}$')
    axs[1, 1].axis('off')
    obj_true_2 = min_max_norm(obj_true_2)
    axs[1, 1].imshow(obj_true_2, cmap='gray')

    obj_pred_1 = min_max_norm(obj_pred_1)
    psnr = ssim(obj_pred_1, obj_true_1, data_range=1)
    axs[0, 2].set_title(r'$\boldsymbol{\hat{S_1}}$, SSIM=%.2f'%psnr)
    axs[0, 2].axis('off')
    axs[0, 2].imshow(obj_pred_1, cmap='gray')

    obj_pred_2 = min_max_norm(obj_pred_2)
    psnr = ssim(obj_pred_2, obj_true_2, data_range=1)
    axs[1, 2].set_title(r'$\boldsymbol{\hat{S_2}}$, SSIM=%.2f'%psnr)
    axs[1, 2].axis('off')
    axs[1, 2].imshow(obj_pred_2, cmap='gray')

    axbig = fig.add_subplot(gs[:, -1])
    axbig.set_title(r'$\boldsymbol{\hat{B}}$')
    axbig.axis('off')
    axbig.imshow(ref_pred, cmap='gray')

    fig.tight_layout()
    plt.savefig(f'{save_dir}/compare_{str(i).zfill(3)}.png', transparent=True)
    plt.close()


def inference(args, mode='valid', size=256, seed=0):
    """
    mode(str): 'train', 'valid' or 'test'
    size(int): image size
    seed(int): random seed of diffusion model
    """
    random.seed(seed)
    with open(args.configs, 'r') as f:
        configs = yaml.safe_load(f)
    data_configs = configs['data_settings']
    model_configs = configs['model_settings']

    model = Diffusion_UNet(model_configs).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_ckpt, map_location='cuda:0'), strict=False)
    model.eval()
    sampler = DDIM_Sampler(model, configs['ddpm_settings'], ddim_sampling_steps=50).cuda()

    # load corresponding data
    if mode=='train':
        data = NanoCT_Pair_Dataset(data_configs['train_data_dir'], img_size=size)
        n_samples = 10

    elif mode=='valid':
        val_path = './valid_data_n/data2/original/'
        val_folders = sorted(os.listdir('./valid_data_n/data2/gt_dref/'))
        n_samples = len(val_folders)

    # model inference
    with torch.no_grad():
        for i in range(n_samples):

            if  mode=='train':
                input_imgs, ref_img = data[random.randint(0, 9999)]
                ref_img = ref_img.squeeze()
                obj_true_1 = (input_imgs[0]/(ref_img+1e-4)).numpy()
                obj_true_2 = (input_imgs[1]/(ref_img+1e-4)).numpy()

            elif mode=='valid':
                val_files = [_ for _ in os.listdir(val_path+val_folders[i]+'/') if _.endswith('tif')]
                rnd_id = random.randint(0, len(val_files)-2)
                input_1 = np.array(Image.open('%s%s/%s'%(val_path, val_folders[i], val_files[rnd_id])).resize((size, size)))
                input_2 = np.array(Image.open('%s%s/%s'%(val_path, val_folders[i], val_files[rnd_id+1])).resize((size, size)))
                brightness = max(input_1.max(), input_2.max())
                input_1 = torch.from_numpy(input_1/brightness).unsqueeze(0).float()
                input_2 = torch.from_numpy(input_2/brightness).unsqueeze(0).float()
                input_imgs = torch.cat([input_1, input_2], dim=0)
                obj_true_1 = np.array(Image.open('./valid_data_n/data2/gt_dref/%s/%s'%(val_folders[i], val_files[rnd_id])).resize((size, size)))
                obj_true_2 = np.array(Image.open('./valid_data_n/data2/gt_dref/%s/%s'%(val_folders[i], val_files[rnd_id+1])).resize((size, size)))
                ref_img = input_1.squeeze()/torch.from_numpy(obj_true_1) 

            torch.manual_seed(seed)
            noise = torch.randn(size=[1, 1, size, size], device='cuda:0')
            pred = sampler(input_imgs.view(1, 2, size, size).cuda(), noise).squeeze().cpu().numpy()
            pred = pred * (0.7/pred.mean())
            
            obj_pred_1 = input_imgs[0].numpy()/pred
            obj_pred_2 = input_imgs[1].numpy()/pred
            obj_pred_1 = np.clip(obj_pred_1, obj_pred_1[:, :-30].min(), obj_pred_1[:, :-30].max())
            obj_pred_2 = np.clip(obj_pred_2, obj_pred_2[:, :-30].min(), obj_pred_2[:, :-30].max())
            plot_2x3(input_imgs, obj_pred_1, obj_pred_2, obj_true_1, obj_true_2, pred, i, args.img_save_dir)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    os.makedirs(args.img_save_dir, exist_ok=True)
    inference(args)