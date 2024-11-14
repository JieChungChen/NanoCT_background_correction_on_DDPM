import os
import torch
import numpy as np
import glob
from tqdm import tqdm
import random
import matplotlib as mpl
from PIL import Image
from torchvision.utils import save_image, make_grid
mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt

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
    inputs_max = input_imgs.view(n_samples, size**2).max(dim=1).values.view(-1, 1, 1, 1)
    input_imgs = input_imgs/inputs_max

    sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
    for i in range(n_samples):
        noisy_img = torch.randn(size=[1, 1, size, size], device=args.device)
        pred = sampler(input_imgs[i].view(1, 1, size, size).to(args.device), noisy_img).squeeze().cpu().numpy()
        obj_pred = input_imgs[i, 0]/pred
        plot2x3(input_imgs[i, 0], noisy_img, pred, ref_imgs[i, 0], obj_pred, dref_imgs[i, 0], 'trn', i)


def model_eval_for_val(args, model=None, seed=29):
    size = args.img_size
    val_path = './valid_data_n/data2/gt_dref/'
    val_folders = sorted(os.listdir(val_path))

    for i, folder in enumerate(val_folders):
        val_files = [_ for _ in os.listdir(val_path+folder+'/') if _.endswith('tif')]
        random.seed(2024)
        rnd_id = random.randint(0, len(val_files)-1)
        raw_img = Image.open('./valid_data_n/data2/original/%s/%s'%(folder, val_files[rnd_id])).resize((size, size))
        raw_img = np.array(raw_img)
        dref_truth = Image.open('./valid_data_n/data2/gt_dref/%s/%s'%(folder, val_files[rnd_id])).resize((size, size))
        dref_truth = np.array(dref_truth)
        ref_truth = (raw_img/dref_truth)
        input_img = torch.Tensor(raw_img/raw_img.max())
    
        sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
        torch.manual_seed(seed)
        noise = torch.randn(size=[1, 1, size, size], device=args.device)
        pred = sampler(input_img.view(1, 1, size, size).to(args.device), noise).squeeze().cpu().numpy()
        obj_pred = input_img/pred
        plot2x3(input_img, noise, pred, ref_truth, obj_pred, dref_truth, 'val', i)


def inference(args, file_path='./tif-no ref/Fr5-b2-60s-m1/', model=None):
    size = args.img_size
    infer_path = sorted(glob.glob("%s/*.tif"%file_path))
    true_ref = np.array(Image.open(infer_path[0]).resize((size, size)))
    print(true_ref.max())
    with torch.no_grad():
        for i, file in enumerate(infer_path):
            raw_img = np.array(Image.open(file).resize((size, size)))
            true_de = np.array(raw_img/(true_ref+1e-2))
            input_img = torch.Tensor(raw_img/raw_img.max())

            sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
            torch.manual_seed(702)
            noise = torch.randn(size=[1, 1, size, size], device=args.device)
            pred = sampler(input_img.view(1, 1, size, size).to(args.device), noise).squeeze().cpu()
            obj_pred = input_img.squeeze()/pred
            obj_pred = torch.clamp(obj_pred, min=obj_pred[:, :-30].min(), max=obj_pred[:, :-30].max())
            save_image(obj_pred,'./tif-no ref/Fr5-b2-60s-m1-result/'+str(i).zfill(2)+'.tif', normalize=True)

            obj_pred = (obj_pred-obj_pred.min())/(obj_pred.max()-obj_pred.min())
            true_de = true_de.clip(true_de[:, :-30].min(), true_de[:, :-30].max())
            true_de = (true_de-true_de.min())/(true_de.max()-true_de.min())
            psnr = -10*torch.log10(((obj_pred-true_de)**2).mean())
            fig = plt.figure()
            plt.subplot(221)
            plt.title('Input img')
            plt.axis('off')
            plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
            plt.subplot(222)
            plt.title('Dref, psnr=%.2f' % psnr)
            plt.axis('off')
            plt.imshow(obj_pred, cmap='gray', vmin=obj_pred.min(), vmax=obj_pred.max())
            plt.subplot(223)
            plt.title('Pred Ref.')
            plt.axis('off')
            plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
            plt.subplot(224)
            plt.title('True Dref')
            plt.axis('off')
            plt.imshow(true_de, cmap='gray', vmin=0, vmax=1)
            fig.tight_layout()
            plt.savefig('./tif-no ref/Fr5-b2-60s-m1-result/compare_'+str(i).zfill(2)+'.png')
            plt.close()


def unpatchify(patch_dir='./tif-no ref/Fr5-b2-60s-m1-result', n_rows=5):
    files = sorted(glob.glob("%s/*.tif"%patch_dir))
    imgs = []
    for f in files:
        imgs.append(np.array(Image.open(f).convert('L')))
    imgs = np.array(imgs).astype(float)

    size = 256
    n_cols = len(imgs)//n_rows
    mosaic = np.zeros((n_rows*size, n_cols*size))
    img_id = 0
    for i in reversed(range(n_rows)):
        for j in range(n_cols):
            im = imgs[img_id]
            im = im/im.max()
            if i<n_rows-1 or j>0:
                bound_avg = im[:, 0].mean() if j>0 else im[-1, :].mean()
                prev_bound_avg = imgs[img_id-1][:, -1].mean() if j>0 else imgs[img_id-n_cols][0, :].mean()
                bound_diff = prev_bound_avg-bound_avg
                im = im+bound_diff
            imgs[img_id] = im
            mosaic[i*size:(i+1)*size, j*size:(j+1)*size] = im
            img_id += 1

    plt.imshow(mosaic, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('pred.png',bbox_inches='tight', pad_inches=0.0, dpi=1200)
    plt.close()


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
    psnr = -10*torch.log10(((obj_pred-dref_truth)**2).mean())
    obj_pred = obj_pred.clip(obj_pred[:, :-25].min(), obj_pred[:, :-25].max())
    obj_pred = (obj_pred-obj_pred.min())/(obj_pred.max()-obj_pred.min())
    dref_truth = (dref_truth-dref_truth.min())/(dref_truth.max()-dref_truth.min())
    plt.title('dref, psnr=%.2f'%psnr)
    plt.axis('off')
    plt.imshow(obj_pred, cmap='gray')
    plt.subplot(236)
    plt.title('dref gt')
    plt.imshow(dref_truth, cmap='gray')
    plt.axis('off')
    fig.tight_layout()
    plt.savefig('figures/'+figname+'_'+str(fig_id).zfill(3)+'.png')


if __name__ == '__main__':
    unpatchify()