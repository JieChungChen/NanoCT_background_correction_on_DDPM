import numpy as np
import torch
import glob
from tifffile import imread
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from PIL import Image

from ddpm.model import Diffusion_UNet
from ddpm.diffusion import DDIM_Sampler


def inference():
    model = Diffusion_UNet(use_torch_attn=False).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('checkpoints/ckpt_step=77343.pt', map_location='cuda:0'), strict=False)
    model.eval()
    file_path='./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15.tif'
    n_cols=19
    size = 256
    raw_imgs = np.flipud(imread(file_path))
    sampler = DDIM_Sampler(model, 1e-4, 2e-2, 'quad', 1000).cuda()
    with torch.no_grad():
        imgs = []
        for i, raw in enumerate(raw_imgs):
            input_img = torch.Tensor(raw/raw.max())
            torch.manual_seed(29)
            noise = torch.randn(size=[1, 1, size, size], device='cuda:0')
            pred = sampler(input_img.view(1, 1, size, size).cuda(), noise).squeeze().cpu()
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
            plt.savefig('./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15/compare_'+str(i).zfill(3)+'.png')
            plt.close()

        factor = imgs[8].max()-imgs[8].min()
        for i, im in enumerate(imgs):
            im = (im-im.min())/factor
            # mean_diff, count = 0, 0
            # if i>0:
            #     if i//n_cols>0:
            #         b_avg = im[-80:, :].median()
            #         prev_b_avg = imgs[i-n_cols][:80, :].median()
            #         mean_diff += prev_b_avg-b_avg
            #         count += 1
            #     if i%n_cols>0:
            #         b_avg = im[:, :80].median()
            #         prev_b_avg = imgs[i-1][:, -80:].median()  
            #         mean_diff += prev_b_avg-b_avg
            #         count += 1
            #     im = im+mean_diff/count
            # imgs[i] = im
            save_image(im,'./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15/'+str(i).zfill(3)+'.tif', normalize=False)


def mosaic(patch_dir='./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15-result', n_rows=19):
    files = sorted(glob.glob("%s/*.tif"%patch_dir))
    # imgs = np.flipud(imread('./tif-no ref/mosaic4-Dr.n-hum-D-b4-20s-m15x15-dref.tif'))
    imgs = []
    for f in files:
        imgs.append(np.array(Image.open(f).convert('L')))
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
                    b_avg = np.mean(im[30:-30, :15])
                    prev_b_avg = np.mean(imgs[img_id-1][30:-30, -15:])
                    diff += (prev_b_avg-b_avg)
                    count += 1
                if i<(n_rows-1):
                    b_avg = np.mean(im[-15:, :-30])
                    prev_b_avg = np.mean(imgs[img_id-n_cols][:15, :-30])
                    diff += prev_b_avg-b_avg
                    count += 1
                im = im + (diff/count)
            imgs[img_id] = im
            mosaic[i*size:(i+1)*size, j*size:(j+1)*size] = im
            img_id += 1 

    # for i, f in enumerate(files):
    #     im = Image.fromarray(imgs[i])
    #     im.save(f)

    plt.imshow(mosaic, cmap='gray', vmax=mosaic.max(), vmin=mosaic.min())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('pred.tif',bbox_inches='tight', pad_inches=0.0, dpi=1200)
    plt.close()

# inference()
mosaic()