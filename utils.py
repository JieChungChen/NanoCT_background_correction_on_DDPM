import os
import torch
import numpy as np
import glob
from tqdm import tqdm
import random
import matplotlib as mpl
from PIL import Image
from torchvision.utils import save_image
mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import DDPM_Sampler, DDIM_Sampler


def check_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = local_rank = world_size = -1
    is_distributed = world_size != -1
    return rank, local_rank, world_size, is_distributed


def model_eval(args, n_samples=8, model=None):
    with torch.no_grad():
        size = args.img_size
        dref_files = sorted(glob.glob("%s/dref/*.tif"%args.data_dir))
        ref_files = sorted(glob.glob("%s/ref/*.tif"%args.data_dir))
        dref_imgs, ref_imgs = [], []
        dref_rnd_choose = np.random.choice(len(dref_files), n_samples, replace=False)
        ref_rnd_choose = np.random.choice(len(ref_files), n_samples, replace=False)
        for i in tqdm(dref_rnd_choose, dynamic_ncols=True, desc='load dref images'):
            raw_dref = Image.open(dref_files[i]).resize((size, size))
            dref_imgs.append(np.array(raw_dref))
        for i in tqdm(ref_rnd_choose, dynamic_ncols=True, desc='load ref images'):
            raw_ref = Image.open(ref_files[i]).resize((size, size))
            ref_imgs.append(np.array(raw_ref))
        dref_imgs = torch.Tensor(np.array(dref_imgs)).unsqueeze(1)
        ref_imgs = torch.Tensor(np.array(ref_imgs)).unsqueeze(1)
        input_imgs = [dref_imgs[i]*ref_imgs[i] for i in range(n_samples)]
        input_imgs = torch.concatenate(input_imgs, dim=0).unsqueeze(1)
        pair_wise_maximum = input_imgs.view(n_samples, size**2).max(dim=1).values.view(-1, 1, 1, 1)
        input_imgs = input_imgs/pair_wise_maximum
        ref_imgs = ref_imgs/pair_wise_maximum

        if model is None:
            model = Diffusion_UNet().to(args.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=args.device), strict=False)
            print("Model weight load down.")

        model.eval()
        sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
        for i in range(n_samples):
            noisy_img = torch.randn(size=[1, 1, size, size], device=args.device)
            pred = sampler(input_imgs[i].view(1, 1, size, size).to(args.device), noisy_img).squeeze().cpu().numpy()
            obj_pred = input_imgs[i, 0]/pred
            plot2x3(input_imgs[i, 0], noisy_img, pred, ref_imgs[i, 0], obj_pred, dref_imgs[i, 0], 'trn', i)


def model_eval_for_val(args, model=None):
    val_path = './valid_data_n/data2/gt_dref/'
    size = args.img_size
    val_folders = sorted(os.listdir(val_path))
    with torch.no_grad():
        if model is None:
            model = Diffusion_UNet().to(args.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=args.device), strict=False)
            print("Model weight load down.")
        model.eval()
        for i, folder in enumerate(val_folders):
            val_files = [_ for _ in os.listdir(val_path+folder+'/') if _.endswith('tif')]
            random.seed(2024)
            rnd_id = random.randint(0, len(val_files)-1)
            raw_img = Image.open('./valid_data_n/data2/original/%s/%s'%(folder, val_files[rnd_id])).resize((size, size))
            raw_img = np.array(raw_img)
            dref_truth = Image.open('./valid_data_n/data2/gt_dref/%s/%s'%(folder, val_files[rnd_id])).resize((size, size))
            dref_truth = np.array(dref_truth)
            ref_truth = (raw_img/dref_truth)/raw_img.max()
            input_img = torch.Tensor(raw_img/raw_img.max())
        
            sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
            noise = torch.randn(size=[1, 1, size, size], device=args.device)
            pred = sampler(input_img.view(1, 1, size, size).to(args.device), noise).squeeze().cpu().numpy()
            obj_pred = input_img/pred
            plot2x3(input_img, noise, pred, ref_truth, obj_pred, dref_truth, 'val', i)


def inference(args, file_path='./tif-no ref/Fr5-b2-60s-m6/', model=None):
    size = args.img_size
    infer_path = sorted(glob.glob("%s/*.tif"%file_path))
    with torch.no_grad():
        if model is None:
                model = Diffusion_UNet().to(args.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=args.device), strict=False)
                print("Model weight load down.")
        model.eval()
        for i, file in enumerate(infer_path):
            raw_img = Image.open(file).resize((size, size))
            raw_img = np.array(raw_img)
            input_img = torch.Tensor(raw_img/raw_img.max())

            sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
            noise = torch.randn(size=[1, 1, size, size], device=args.device)
            pred = sampler(input_img.view(1, 1, size, size).to(args.device), noise).squeeze().cpu()
            obj_pred = input_img.squeeze()/pred
            save_image(pred, './tif-no ref/ref/Fr5-b2-60s-m6/pred-ref-%s.tif'%str(i+1).zfill(3), normalize=True)
            save_image(obj_pred, './tif-no ref/dref/Fr5-b2-60s-m6/pred-dref-%s.tif'%str(i+1).zfill(3), normalize=True, 
                       value_range=(obj_pred[:, :-15].min(), obj_pred[:, :-15].max()))


def model_eval_64x64patch(args, n_samples=8, model=None):
    with torch.no_grad():
        dref_files = sorted(glob.glob("%s/dref/*.tif"%args.data_dir))
        ref_files = sorted(glob.glob("%s/ref/*.tif"%args.data_dir))
        dref_imgs, ref_imgs = [], []
        dref_rnd_choose = np.random.choice(len(dref_files), n_samples, replace=False)
        ref_rnd_choose = np.random.choice(len(ref_files), n_samples, replace=False)
        for i in tqdm(dref_rnd_choose, dynamic_ncols=True, desc='load dref images'):
            raw_dref = Image.open(dref_files[i])
            dref_imgs.append(np.array(raw_dref))
        for i in tqdm(ref_rnd_choose, dynamic_ncols=True, desc='load ref images'):
            raw_ref = Image.open(ref_files[i])
            ref_imgs.append(np.array(raw_ref))
        dref_imgs = torch.Tensor(np.array(dref_imgs)).unsqueeze(1)
        ref_imgs = torch.Tensor(np.array(ref_imgs)).unsqueeze(1)
        input_imgs = [dref_imgs[i]*ref_imgs[i] for i in range(n_samples)]
        input_imgs = torch.concatenate(input_imgs, dim=0).unsqueeze(1)
        pair_wise_maximum = input_imgs.view(n_samples, 512**2).max(dim=1).values.view(-1, 1, 1, 1)
        input_imgs = input_imgs/pair_wise_maximum
        ref_imgs = ref_imgs/pair_wise_maximum

        if model is None:
            model = Diffusion_UNet().to(args.device)
            # model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=args.device), strict=False)
            print("Model weight load down.")

        model.eval()
        sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
        for i in range(n_samples):
            noisy_img = torch.randn(size=[1, 1, 64, 64], device=args.device).repeat(64, 1, 1, 1)
            input_patches = torch.nn.functional.unfold(input_imgs[i].view(1, 1, 512, 512), kernel_size=64, stride=64)
            input_patches = input_patches.view(1, 64, 64, -1).permute(3, 0, 1, 2)
            pred = sampler(input_patches.to(args.device), noisy_img).cpu()
            obj_pred = (input_patches/pred).squeeze().numpy()
            input_patches = unpatchify(input_patches.squeeze().numpy())
            pred = unpatchify(pred.squeeze().numpy())
            obj_pred = unpatchify(obj_pred)
            plot2x3(input_patches, noisy_img[0], pred.squeeze(), ref_imgs[i, 0], obj_pred, dref_imgs[i, 0], 'trn', i)


def plot2x3(input_img, noise, pred, ref_truth, obj_pred, dref_truth, figname, fig_id=0):
    fig = plt.figure()
    plt.subplot(231)
    plt.title('input img')
    plt.axis('off')
    plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
    plt.subplot(234)
    plt.title('noise')
    plt.axis('off')
    plt.imshow(noise.squeeze().cpu().numpy(), cmap='gray')
    plt.subplot(232)
    plt.title('ref pred')
    plt.axis('off')
    plt.imshow(pred, cmap='gray', vmin=pred.min(), vmax=pred.max())
    plt.subplot(235)
    plt.title('ref gt')
    plt.axis('off')
    plt.imshow(ref_truth, cmap='gray', vmin=ref_truth.min(), vmax=ref_truth.max())
    plt.subplot(233)
    plt.title('input-pred')
    plt.axis('off')
    plt.imshow(obj_pred, cmap='gray', vmin=obj_pred[:, :-10].min(), vmax=obj_pred[:, :-10].max())
    plt.subplot(236)
    plt.title('dref gt')
    plt.imshow(dref_truth, cmap='gray', vmin=dref_truth.min(), vmax=dref_truth.max())
    plt.axis('off')
    fig.tight_layout()
    plt.savefig('figures/'+figname+'_'+str(fig_id).zfill(3)+'.png')
    

def unpatchify(patches):
    n_col_row = int(patches.shape[0]**0.5)
    p_size = patches.shape[-1]
    size = p_size*n_col_row
    img = np.zeros((size, size))
    p_count = 0
    for i in range(n_col_row):
        for j in range(n_col_row):
            img[i*p_size:(i+1)*p_size, j*p_size:(j+1)*p_size] = patches[p_count]
            p_count += 1
    return img