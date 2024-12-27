import torch
import os, glob, random
import numpy as np
from tifffile import imread
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from data_preprocess import NanoCT_Pair_Dataset
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import DDIM_Sampler


def inference_train():
    model = Diffusion_UNet(use_torch_attn=False, input_ch=3).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('ckpts_pair/ddpm_pair_160K.pt', map_location='cuda:0'), strict=False)
    model.eval()

    size = 256
    training_data = NanoCT_Pair_Dataset('./training_data_n', img_size=256)
    
    sampler = DDIM_Sampler(model, 1e-4, 2e-2, 'linear', 1000, ddim_sampling_steps=50).cuda()
    with torch.no_grad():
        for i in range(10):
            input_imgs, ref_img = training_data[i*100]
            ref_img = ref_img.squeeze()
            torch.manual_seed(29)
            noise = torch.randn(size=[1, 1, size, size], device='cuda:0')
            pred = sampler(input_imgs.view(1, 2, size, size).cuda(), noise).squeeze().cpu()
            obj_pred_1 = input_imgs[0]/pred
            obj_pred_2 = input_imgs[1]/pred
            obj_pred_1 = torch.clamp(obj_pred_1, min=obj_pred_1[:, :-40].min(), max=obj_pred_1[:, :-40].max())
            obj_pred_2 = torch.clamp(obj_pred_2, min=obj_pred_2[:, :-40].min(), max=obj_pred_2[:, :-40].max())
            obj_pred_1 = (obj_pred_1-obj_pred_1.min())/(obj_pred_1.max()-obj_pred_1.min())
            obj_pred_2 = (obj_pred_2-obj_pred_2.min())/(obj_pred_2.max()-obj_pred_2.min())

            plot_2x4(input_imgs, pred, ref_img, obj_pred_1, obj_pred_2, 
                     (input_imgs[0]/ref_img).numpy(), (input_imgs[1]/ref_img).numpy(), i)


def inference_val():
    model = Diffusion_UNet(use_torch_attn=False, input_ch=3).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('ckpts_pair/ddpm_pair_160K.pt', map_location='cuda:0'), strict=False)
    model.eval()

    size = 256
    val_path = './valid_data_n/data2/original/'
    val_folders = sorted(os.listdir('./valid_data_n/data2/gt_dref/'))
    
    sampler = DDIM_Sampler(model, 1e-4, 2e-2, 'linear', 1000, ddim_sampling_steps=50).cuda()
    with torch.no_grad():
        for i, folder in enumerate(val_folders):
            val_files = [_ for _ in os.listdir(val_path+folder+'/') if _.endswith('tif')]
            rnd_id = random.randint(0, len(val_files)-2)
            input_1 = np.array(Image.open('%s%s/%s'%(val_path, folder, val_files[rnd_id])).resize((size, size)))
            input_2 = np.array(Image.open('%s%s/%s'%(val_path, folder, val_files[rnd_id+1])).resize((size, size)))
            input_1 = torch.from_numpy(input_1/input_1.max()).unsqueeze(0).float()
            input_2 = torch.from_numpy(input_2/input_2.max()).unsqueeze(0).float()
            input_imgs = torch.cat([input_1, input_2], dim=0)
            obj_true_1 = np.array(Image.open('./valid_data_n/data2/gt_dref/%s/%s'%(folder, val_files[rnd_id])).resize((size, size)))
            obj_true_2 = np.array(Image.open('./valid_data_n/data2/gt_dref/%s/%s'%(folder, val_files[rnd_id+1])).resize((size, size)))
            ref_img = input_1.squeeze()/torch.from_numpy(obj_true_1)
            torch.manual_seed(0)
            noise = torch.randn(size=[1, 1, size, size], device='cuda:0')
            pred = sampler(input_imgs.view(1, 2, size, size).cuda(), noise).squeeze().cpu()
            obj_pred_1 = input_imgs[0]/pred
            obj_pred_2 = input_imgs[1]/pred
            obj_pred_1 = torch.clamp(obj_pred_1, min=obj_pred_1[:, :-30].min(), max=obj_pred_1[:, :-30].max())
            obj_pred_2 = torch.clamp(obj_pred_2, min=obj_pred_2[:, :-30].min(), max=obj_pred_2[:, :-30].max())
            obj_pred_1, obj_pred_2 = min_max_norm(obj_pred_1), min_max_norm(obj_pred_2)

            plot_2x4(input_imgs, pred, ref_img, obj_pred_1, obj_pred_2, obj_true_1, obj_true_2, i)

def inference_test(folder='./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15', compare=True):
    model = Diffusion_UNet(use_torch_attn=False, input_ch=3).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('ckpts_pair/ddpm_pair_130K.pt', map_location='cuda:0'), strict=False)
    model.eval()

    file_path='./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15.tif'
    size = 256
    raw_imgs = np.flipud(imread(file_path))
    # tif_files = sorted(glob.glob("%s/*.tif"%folder))
    # raw_imgs = [np.array(Image.open(f).resize((size, size))) for f in tif_files]

    sampler = DDIM_Sampler(model, 1e-4, 2e-2, 'linear', 1000).cuda()
    with torch.no_grad():
        imgs = []
        for i in range(len(raw_imgs)-1):
            print('img: %d'%i)
            input_1 = torch.Tensor(raw_imgs[i]/raw_imgs[i].max()).unsqueeze(0).float()
            input_2 = torch.Tensor(raw_imgs[i+1]/raw_imgs[i+1].max()).unsqueeze(0).float()
            input_imgs = torch.cat([input_1, input_2], dim=0)
            # torch.manual_seed(29)
            noise = torch.randn(size=[1, 1, size, size], device='cuda:0')
            pred = sampler(input_imgs.view(1, 2, size, size).cuda(), noise).squeeze().cpu()
            obj_pred_1 = input_imgs[0]/pred
            obj_pred_2 = input_imgs[1]/pred
            obj_pred_1 = torch.clamp(obj_pred_1, min=obj_pred_1[:, :-30].min(), max=obj_pred_1[:, :-30].max())
            obj_pred_2 = torch.clamp(obj_pred_2, min=obj_pred_2[:, :-30].min(), max=obj_pred_2[:, :-30].max())
            # obj_pred_1, obj_pred_2 = min_max_norm(obj_pred_1), min_max_norm(obj_pred_2)
            imgs.append(obj_pred_1.unsqueeze(2))
            if i == len(raw_imgs)-2:
                imgs.append(obj_pred_2.unsqueeze(2))

            if compare:
                fig = plt.figure(figsize=(6, 4))
                plt.subplot(231)
                plt.title('input img1')
                plt.axis('off')
                plt.imshow(input_imgs[0], cmap='gray', vmin=0, vmax=1)

                plt.subplot(234)
                plt.title('input img2')
                plt.axis('off')
                plt.imshow(input_imgs[1], cmap='gray', vmin=0, vmax=1)

                plt.subplot(232)
                plt.title('pred_ref')
                plt.axis('off')
                plt.imshow(pred, cmap='gray')

                plt.subplot(233)
                plt.title('pred dref1')
                plt.axis('off')
                plt.imshow(obj_pred_1, cmap='gray')

                plt.subplot(236)
                plt.title('pred dref2')
                plt.axis('off')
                plt.imshow(obj_pred_2, cmap='gray')

                fig.tight_layout()
                plt.savefig(folder+'-result/compare/'+str(i).zfill(3)+'.png')
                plt.close()
        
        # imgs = mosaic_norm(imgs, n_rows=19)
        imgs = torch.cat(imgs, dim=2).permute(2, 0, 1)
        g_min = imgs.min()
        factor = imgs.max()-imgs.min()
        for i, im in enumerate(imgs):
            im = (im-g_min)/factor
            im = Image.fromarray(im.numpy())
            im.save(folder+'-result/'+str(i).zfill(3)+'.tif')
            # save_image(im, folder+'-result/'+str(i).zfill(3)+'.tif', normalize=False)

def min_max_norm(x):
    return (x-x.min())/(x.max()-x.min())

def calc_psnr(x, y):
    return -10*torch.log10(((x-y)**2).mean())

def plot_2x4(input_imgs, pred, ref_img, obj_pred_1, obj_pred_2, obj_true_1, obj_true_2, i):
    fig = plt.figure(figsize=(9, 4))
    plt.subplot(241)
    plt.title('input img1')
    plt.axis('off')
    plt.imshow(input_imgs[0], cmap='gray', vmin=0, vmax=1)

    plt.subplot(245)
    plt.title('input img2')
    plt.axis('off')
    plt.imshow(input_imgs[1], cmap='gray', vmin=0, vmax=1)

    plt.subplot(242)
    plt.title('pred_ref')
    plt.axis('off')
    plt.imshow(pred, cmap='gray')

    plt.subplot(246)
    plt.title('true_ref')
    plt.axis('off')
    plt.imshow(ref_img, cmap='gray')

    plt.subplot(247)
    plt.title('true dref1')
    plt.axis('off')
    obj_true_1 = min_max_norm(obj_true_1)
    plt.imshow(obj_true_1, cmap='gray')

    plt.subplot(248)
    plt.title('true dref2')
    plt.axis('off')
    obj_true_2 = min_max_norm(obj_true_2)
    plt.imshow(obj_true_2, cmap='gray')

    plt.subplot(243)
    plt.title('pred dref1, SSIM=%.1f'%ssim((obj_pred_1.numpy()*255).astype('uint8'), (obj_true_1*255).astype('uint8'), multichannel=False))
    plt.axis('off')
    plt.imshow(obj_pred_1, cmap='gray')

    plt.subplot(244)
    plt.title('pred dref2, SSIM=%.1f'%ssim((obj_pred_2.numpy()*255).astype('uint8'), (obj_true_2*255).astype('uint8'), multichannel=False))
    plt.axis('off')
    plt.imshow(obj_pred_2, cmap='gray')

    fig.tight_layout()
    plt.savefig('./figures/compare_'+str(i).zfill(3)+'.png')
    plt.close()

def mosaic_norm(imgs, n_rows):
    n_cols = len(imgs)//n_rows
    img_id = 0
    for i in range(n_rows):
        for j in range(n_cols):
            im = imgs[img_id]
            if i!=0 or j!=0:
                diff, count = 0, 0
                if j>0:
                    b_avg = im[30:-30, :30].mean()
                    prev_b_avg = imgs[img_id-1][30:-30, -30:].mean()
                    diff += (prev_b_avg-b_avg)
                    count += 1
                if i>0:
                    b_avg = im[-30:, :-30].mean()
                    prev_b_avg = imgs[img_id-n_cols][:30, :-30].mean()
                    diff += prev_b_avg-b_avg
                    count += 1
                im = im + (diff/count)
            imgs[img_id] = im.reshape(256, 256, 1)
            img_id += 1 
    return imgs


def mosaic(patch_dir='./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15-result', n_rows=19, raw_tif=False):
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
                diff, count = 0, 0
                if i==n_rows-1 or j>0:
                    b_avg = np.mean(im[30:-30, 5:])
                    prev_b_avg = np.mean(imgs[img_id-1][30:-30, -5:])
                    diff += (prev_b_avg-b_avg)
                    count += 1
                if i<(n_rows-1):
                    b_avg = np.mean(im[-5:, :-30])
                    prev_b_avg = np.mean(imgs[img_id-n_cols][:5, :-30])
                    diff += prev_b_avg-b_avg
                    count += 1
                if (diff/count)>0.02:
                    im = im + (diff/count)
            imgs[img_id] = im
            mosaic[i*size:(i+1)*size, j*size:(j+1)*size] = im
            img_id += 1 

    if raw_tif:
        for i, f in enumerate(files):
            im = Image.fromarray(imgs[i])
            im.save(f)

    mosaic = Image.fromarray(mosaic)
    mosaic.save('pred.tif')

# inference_test()
mosaic()