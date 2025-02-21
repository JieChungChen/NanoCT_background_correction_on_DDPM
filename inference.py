import torch
import torch.nn.functional as F
import os, glob, random, yaml
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from data_preprocess import NanoCT_Pair_Dataset
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import DDIM_Sampler
from utils import min_max_norm, calc_psnr


def plot_2x3(input_imgs, obj_pred_1, obj_pred_2, obj_true_1, obj_true_2, i):
    fig = plt.figure(figsize=(6.5, 4))
    plt.subplot(231)
    plt.title('input img 1')
    plt.axis('off')
    plt.imshow(input_imgs[0], cmap='gray')

    plt.subplot(234)
    plt.title('input img 2')
    plt.axis('off')
    plt.imshow(input_imgs[1], cmap='gray')

    plt.subplot(232)
    plt.title('true sample 1')
    plt.axis('off')
    obj_true_1 = min_max_norm(obj_true_1)
    plt.imshow(obj_true_1, cmap='gray')

    plt.subplot(235)
    plt.title('true sample 2')
    plt.axis('off')
    obj_true_2 = min_max_norm(obj_true_2)
    plt.imshow(obj_true_2, cmap='gray')

    plt.subplot(233)
    plt.title('prediction, SSIM=%.1f'%ssim((obj_pred_1*255).astype('uint8'), (obj_true_1*255).astype('uint8'), multichannel=False))
    plt.axis('off')
    plt.imshow(obj_pred_1, cmap='gray')

    plt.subplot(236)
    plt.title('prediction, SSIM=%.1f'%ssim((obj_pred_2*255).astype('uint8'), (obj_true_2*255).astype('uint8'), multichannel=False))
    plt.axis('off')
    plt.imshow(obj_pred_2, cmap='gray')

    fig.tight_layout()
    plt.savefig('./figures/compare_'+str(i).zfill(3)+'.png')
    plt.close()


def inference(mode='valid', size=256, seed=1, compare=True):
    """
    mode(str): 'train', 'valid' or 'test'
    size(int): image size
    seed(int): random seed of diffusion model
    """
    with open('configs/ddpm_pair_base.yml', 'r') as f:
        configs = yaml.safe_load(f)
    model_configs = configs['model_settings']

    model = Diffusion_UNet(model_configs).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('checkpoints/ddpm_pair_160K.pt', map_location='cuda:0'), strict=False)
    model.eval()
    sampler = DDIM_Sampler(model, configs['ddpm_settings'], ddim_sampling_steps=50).cuda()

    # load corresponding data
    if mode=='train':
        data = NanoCT_Pair_Dataset('./training_data_n', img_size=size)
        n_samples = 10

    elif mode=='valid':
        val_path = './valid_data_n/data2/original/'
        val_folders = sorted(os.listdir('./valid_data_n/data2/gt_dref/'))
        n_samples = len(val_folders)

    elif mode=='test':
        folder='./tif-no ref/Fr5-b2-60s-m1'
        tif_files = sorted(glob.glob("%s/*.tif"%folder))
        raw_imgs = [np.array(Image.open(f).resize((size, size)))+10 for f in tif_files]
        n_samples = len(raw_imgs)-1

    # model inference
    with torch.no_grad():
        for i in range(n_samples):

            if  mode=='train':
                input_imgs, ref_img = data[i*100]
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
            
            elif mode=='test':
                input_1 = torch.Tensor(raw_imgs[i]/200).unsqueeze(0).float()
                input_2 = torch.Tensor(raw_imgs[i+1]/200).unsqueeze(0).float()
                input_imgs = torch.cat([input_1, input_2], dim=0)

            torch.manual_seed(seed)
            noise = torch.randn(size=[1, 1, size, size], device='cuda:0')
            pred = sampler(input_imgs.view(1, 2, size, size).cuda(), noise).squeeze().cpu().numpy()

            obj_pred_1 = input_imgs[0].numpy()/pred
            obj_pred_2 = input_imgs[1].numpy()/pred
            obj_pred_1 = np.clip(obj_pred_1, obj_pred_1[:, :-30].min(), obj_pred_1[:, :-30].max())
            obj_pred_2 = np.clip(obj_pred_2, obj_pred_2[:, :-30].min(), obj_pred_2[:, :-30].max())

            if mode == 'valid' or 'test':
                obj_pred_1, obj_pred_2 = min_max_norm(obj_pred_1), min_max_norm(obj_pred_2)

            if mode=='test':
                im = Image.fromarray(obj_pred_1)
                im.save(folder+'-result/'+str(i).zfill(3)+'.tif')
                # im_pred = Image.fromarray(pred)
                # im_pred.save(folder+'-result/temp/pred_'+str(i+1).zfill(3)+'.tif')
                if i == (n_samples-1):
                    im = Image.fromarray(obj_pred_2)
                    im.save(folder+'-result/'+str(i+1).zfill(3)+'.tif')

            if compare:
                plot_2x3(input_imgs, obj_pred_1, obj_pred_2, obj_true_1, obj_true_2, i)
                plt.imshow(pred, cmap='gray', vmin=pred.min(), vmax=pred.max())
                plt.savefig(f'./figures/pred_{i+1}.png') 
                plt.close()


def inference_test(folder='./tif-no ref/mam_amber_m02', size=256):
    with open('configs/ddpm_pair_v3.yml', 'r') as f:
        configs = yaml.safe_load(f)
    model_configs = configs['model_settings']
    model = Diffusion_UNet(model_configs).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('checkpoints/ddpm_pair_v3_310K.pt', map_location='cuda:0'), strict=False)
    sampler = DDIM_Sampler(model, configs['ddpm_settings'], ddim_sampling_steps=50).cuda()
    model.eval()

    file_path='./tif-no ref/mam_amber_m02_raw.tif'
    raw_imgs = np.flipud(imread(file_path).transpose((1, 2, 0)))
    raw_imgs = raw_imgs.transpose((2, 0, 1))

    ref = np.array(Image.open("valid_data_n/data2/original/20230301_scarab1-b2-40s-20.3-m3/20230301_scarab1-b2-40s-20.3-m3_0001.tif").resize((size, size)))
    raw_std, ref_std = raw_imgs[0].std(), ref.std()
    raw_mean, ref_mean = ref.mean(), (raw_imgs[0]*(ref_std/raw_std)).mean()
    factor = ref_std/raw_std
    raw_imgs = raw_imgs*factor
    print((raw_imgs[0]).min(), (raw_imgs[0]).max())
    with torch.no_grad():
        for i in range(len(raw_imgs)-1):
            print('img: %d'%i)
            input_1 = torch.Tensor(raw_imgs[i]/10000).unsqueeze(0).float()
            input_2 = torch.Tensor(raw_imgs[i+1]/10000).unsqueeze(0).float()
            input_1 = F.interpolate(input_1.unsqueeze(0), size=(size, size), mode='bicubic')
            input_2 = F.interpolate(input_2.unsqueeze(0), size=(size, size), mode='bicubic')
            input_imgs = torch.cat([input_1, input_2], dim=1)
            torch.manual_seed(1)
            noise = torch.randn(size=[1, 1, size, size], device='cuda:0')
            pred = sampler(input_imgs.view(1, 2, size, size).cuda(), noise).squeeze().cpu()
            obj_pred_1 = input_imgs[0, 0]/pred
            obj_pred_2 = input_imgs[0, 1]/pred
            im = Image.fromarray(obj_pred_1.numpy())
            im.save(folder+'/'+str(i).zfill(3)+'.tif')
            if i == len(raw_imgs)-2:
                im = Image.fromarray(obj_pred_2.numpy())
                im.save(folder+'/'+str(i+1).zfill(3)+'.tif')


def mosaic(patch_dir='./tif-no ref/mam_amber_m02/', n_rows=27):
    files = sorted(glob.glob("%s/*.tif"%patch_dir))
    # imgs = np.flipud(imread('./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15-dref.tif'))
    imgs = []
    for f in files:
        imgs.append(np.array(Image.open(f)))
    imgs = np.array(imgs)
    print(imgs.shape)

    size = 256
    n_cols = len(imgs)//n_rows
    mosaic = np.zeros((n_rows*size, n_cols*size))
    img_id = 0
    for i in reversed(range(n_rows)):
        for j in range(n_cols):
            im = imgs[img_id]
            if i!=(n_rows-1) or j!=0:
                diff = []
                if i==n_rows-1 or j>0:
                    b_avg = np.mean(im[:, 15:])
                    prev_b_avg = np.mean(imgs[img_id-1][:, -15:])
                    diff.append(prev_b_avg/b_avg)
                if i<(n_rows-1):
                    b_avg = np.mean(im[-15:, :])
                    prev_b_avg = np.mean(imgs[img_id-n_cols][:15, :])
                    diff.append(prev_b_avg/b_avg)

                # im = im*np.mean(diff)

            imgs[img_id] = im
            img_id += 1 
            # im = (im-i_min)/(i_max-i_min)
            # im = (im*255).astype(np.int8)
            # im = Image.fromarray(im)
            # im.save(patch_dir+str(img_id-1).zfill(3)+'.tif')
            mosaic[i*size:(i+1)*size, j*size:(j+1)*size] = im

    mosaic = Image.fromarray(mosaic)
    mosaic.save('pred.tif')


if __name__ == '__main__':
    inference()
    # mosaic()