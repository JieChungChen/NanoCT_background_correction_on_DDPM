import os
import torch
import numpy as np
import glob
from tqdm import tqdm
import random
import matplotlib as mpl
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import save_image
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
    np.random.seed(0)
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
        plot1x5(input_imgs[i, 0], pred, ref_imgs[i, 0], obj_pred, dref_imgs[i, 0], 'trn', i)


def model_eval_for_val(args, model=None, seed=0):
    size = args.img_size
    val_path = './valid_data_n/data2/gt_dref/'
    val_folders = sorted(os.listdir(val_path))

    sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
    for i, folder in enumerate(val_folders):
        val_files = [_ for _ in os.listdir(val_path+folder+'/') if _.endswith('tif')]
        # random.seed(2)
        rnd_id = random.randint(0, len(val_files)-1)
        raw_img = Image.open('./valid_data_n/data2/original/%s/%s'%(folder, val_files[rnd_id])).resize((size, size))
        raw_img = np.array(raw_img)
        dref_truth = Image.open('./valid_data_n/data2/gt_dref/%s/%s'%(folder, val_files[rnd_id])).resize((size, size))
        dref_truth = np.array(dref_truth)
        ref_truth = (raw_img/dref_truth)
        input_img = torch.Tensor(raw_img/raw_img.max())

        # ref_truth = np.array(Image.open("./training_data_n/ref/20230422_test_ref.tif").resize((size, size)))
        # input_img = ref_truth*dref_truth
        # input_img = torch.Tensor(input_img/input_img.max())
        
    
        # torch.manual_seed(seed)
        noise = torch.randn(size=[1, 1, size, size], device=args.device)
        pred = sampler(input_img.view(1, 1, size, size).to(args.device), noise).squeeze().cpu().numpy()
        obj_pred = input_img/pred
        plot1x5(input_img, pred, ref_truth, obj_pred, dref_truth, 'val', i)


def test_inference(args, file_path='./tif-no ref/Fr5-b2-60s-m1', model=None, n_cols=8):
    size = args.img_size
    infer_path = sorted(glob.glob("%s/*.tif"%file_path))
    # true_ref = np.array(Image.open("./tif-no ref/ref2-b4-20s.tif"))
    with torch.no_grad():
        imgs = []
        for i, file in enumerate(infer_path):
            raw_img = (np.array(Image.open(file).resize((size, size))))
            input_img = torch.Tensor(raw_img/raw_img.max())

            sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
            torch.manual_seed(0)
            noise = torch.randn(size=[1, 1, size, size], device=args.device)
            pred = sampler(input_img.view(1, 1, size, size).to(args.device), noise).squeeze().cpu()
            obj_pred = input_img.squeeze()/pred
            obj_pred = torch.clamp(obj_pred, min=obj_pred[:, :-30].min(), max=obj_pred[:, :-30].max())
            imgs.append(obj_pred)

            fig = plt.figure(figsize=(6, 2))
            plt.subplot(131)
            plt.title('input img')
            plt.axis('off')
            plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
            plt.subplot(132)
            plt.title('pred_ref')
            plt.axis('off')
            plt.imshow(pred, cmap='gray')
            plt.subplot(133)
            plt.title('pred_dref')
            plt.axis('off')
            plt.imshow(obj_pred, cmap='gray', vmin=obj_pred.min(), vmax=obj_pred.max())
            fig.tight_layout()
            plt.savefig('./tif-no ref/Fr5-b2-60s-m1-result/compare/'+str(i).zfill(2)+'.png')
            plt.close()

        for i, im in enumerate(imgs):
            mean_diff, count = 0, 0
            if i>0:
                if i//n_cols>0:
                    b_avg = im[-15:, :].median()
                    prev_b_avg = imgs[i-n_cols][:15, :].median()
                    mean_diff += prev_b_avg-b_avg
                    count += 1
                if i%n_cols>0:
                    b_avg = im[:, :15].median()
                    prev_b_avg = imgs[i-1][:, -15:].median()  
                    mean_diff += prev_b_avg-b_avg
                    count += 1
                im = im+mean_diff/count
            imgs[i] = im.reshape(256, 256, 1)
        
        imgs = torch.cat(imgs, dim=2).permute(2, 0, 1)
        g_min = imgs.min()
        factor = imgs.max()-imgs.min()
        for i, im in enumerate(imgs):
            im = (im-g_min)/factor
            save_image(im,'./tif-no ref/Fr5-b2-60s-m1-result/'+str(i).zfill(3)+'.tif', normalize=False)


def val_inference(args, file_path='./valid_data_n/data1/original/20230502_Lm-2-b2-60sm5x4', model=None, n_cols=4):
    size = args.img_size
    infer_path = sorted(glob.glob("%s/*.tif"%file_path))
    with torch.no_grad():
        imgs = []
        for i, file in enumerate(infer_path):
            raw_img = np.array(Image.open(file).resize((size, size)))
            input_img = torch.Tensor(raw_img/raw_img.max())

            sampler = DDIM_Sampler(model, args.beta_1, args.beta_T, args.beta_sche, args.T).to(args.device)
            torch.manual_seed(29)
            noise = torch.randn(size=[1, 1, size, size], device=args.device)
            # if i  in [0, 4, 8]:
            pred = sampler(input_img.view(1, 1, size, size).to(args.device), noise).squeeze().cpu()
            obj_pred = input_img.squeeze()/pred
            obj_pred = torch.clamp(obj_pred, min=obj_pred[:, :-30].min(), max=obj_pred[:, :-30].max())
            resize = Resize([512, 512])
            obj_pred = resize(obj_pred.unsqueeze(0))
            imgs.append(obj_pred)

        factor = imgs[10].max()-imgs[10].min()
        for i, im in enumerate(imgs):
            im = (im-im.min())/factor*0.7+0.3
            if i>0:
                if i//n_cols>0:
                    b_avg = im[-30:, :].median()
                    prev_b_avg = imgs[i-n_cols][:30, :].median()
                else:
                    b_avg = im[:, :30].median()
                    prev_b_avg = imgs[i-1][:, -30:].median()  
                mean_diff = prev_b_avg-b_avg
                im = im+mean_diff
            imgs[i] = im
            save_image(im,'./valid_data_n/data1/pred/'+str(i).zfill(2)+'.tif', normalize=False)


def unpatchify(patch_dir='./tif-no ref/mosaic4-Atl-hum-V-b4-20s-m19x13', n_rows=19):
    files = sorted(glob.glob("%s/*.tif"%patch_dir))
    imgs = []
    for f in files:
        imgs.append(np.array(Image.open(f).convert('L')))
    imgs = np.array(imgs).astype(float)

    size = 256
    n_cols = len(imgs)//n_rows
    mosaic = np.zeros((n_rows*size, n_cols*size))
    img_id = 0
    factor=10
    for i in reversed(range(n_rows)):
        for j in range(n_cols):
            im = imgs[img_id]
            im = (im-im.min())/30
            if i!=n_rows-1 or j!=0:
                diff, count = 0, 0
                if i==n_rows-1 or j>0:
                    b_avg = np.median(im[:, :30])
                    prev_b_avg = np.median(imgs[img_id-1][:, -30:])
                    diff += prev_b_avg-b_avg
                    count += 1
                if i<n_rows-1:
                    b_avg = np.median(im[-30:, 0])
                    prev_b_avg = np.median(imgs[img_id-n_cols][:30, :])
                    diff += prev_b_avg-b_avg
                    count += 1
                im = im+diff/count
            imgs[img_id] = im
            mosaic[i*size:(i+1)*size, j*size:(j+1)*size] = im
            img_id += 1 

    plt.imshow(mosaic, cmap='gray', vmax=mosaic.max(), vmin=mosaic.min())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('pred.png',bbox_inches='tight', pad_inches=0.0, dpi=600)
    plt.close()


def plot1x5(input_img, pred, ref_truth, obj_pred, dref_truth, figname, fig_id):
    fig = plt.figure(figsize=(10, 2))
    plt.subplot(151)
    plt.title('input img')
    plt.axis('off')
    plt.imshow(input_img, cmap='gray', vmin=0, vmax=input_img.max())

    plt.subplot(152)
    plt.title('ref(pred)')
    plt.axis('off')
    plt.imshow(pred, cmap='gray', vmin=pred.min(), vmax=pred.max())

    plt.subplot(153)
    plt.title('ref(truth)')
    plt.axis('off')
    plt.imshow(ref_truth, cmap='gray', vmin=ref_truth.min(), vmax=ref_truth.max())

    obj_pred = obj_pred.clip(obj_pred[:, :-36].min(), obj_pred[:, :-36].max())
    obj_pred = (obj_pred-obj_pred.min())/(obj_pred.max()-obj_pred.min())
    dref_truth = (dref_truth-dref_truth.min())/(dref_truth.max()-dref_truth.min())
    psnr = -10*torch.log10(((obj_pred-dref_truth)**2).mean())
    plt.subplot(154)
    plt.title('dref(pred), psnr=%.2f'%psnr)
    plt.axis('off')
    plt.imshow(obj_pred, cmap='gray')

    plt.subplot(155)
    plt.title('dref(truth)')
    plt.axis('off')
    plt.imshow(dref_truth, cmap='gray')
    fig.tight_layout()
    plt.savefig('figures/'+figname+'_'+str(fig_id).zfill(3)+'.png')


if __name__ == '__main__':
    unpatchify()