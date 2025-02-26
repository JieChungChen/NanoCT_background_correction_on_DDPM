import os, glob
import torch
import numpy as np
from PIL import Image


def check_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = local_rank = world_size = -1
    is_distributed = world_size != -1
    return rank, local_rank, world_size, is_distributed


def min_max_norm(x):
    return (x-x.min())/(x.max()-x.min())


def calc_psnr(x, y):
    return -10*torch.log10(((x-y)**2).mean())


def raw_img_save(patch_dir='./tif-no ref/Fr5-b2-60s-m2'):
    files = sorted(glob.glob("%s/*.tif"%patch_dir))[24:]
    for i, f in enumerate(files):
        im = Image.open(f).resize((256, 256))
        im.save(patch_dir+'-result/'+str(i).zfill(3)+'.tif')


def mosaic(patch_dir='./tif-no ref/mam_amber_m02/', save_dir='figs_temp', n_rows=27, auto_contrast=False):
    """
    reconstruct a mosaic, start from bottom left corner (go right to the end then up)

    patch_dir(str): the folder containing tif patch files of a mosaic.
    save_dir(str): the folder where the result will be saved
    n_rows(int): the number of rows in the mosaic.
    auto_contrast(bool): whether to automatically adjust the contrast of the mosaic.
    """
    files = sorted(glob.glob("%s/*.tif"%patch_dir))
    imgs = []
    for f in files:
        imgs.append(np.array(Image.open(f)))
    imgs = np.array(imgs)
    print(f'n_figures: {imgs.shape[0]}, resolution: {imgs.shape[1:]}')

    size = imgs.shape[1]
    n_cols = len(imgs)//n_rows
    mosaic = np.zeros((n_rows*size, n_cols*size))
    img_id = 0
    for i in reversed(range(n_rows)):
        for j in range(n_cols):
            im = imgs[img_id]

            if auto_contrast:
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
                im = im*np.mean(diff)
                imgs[img_id] = im

            img_id += 1 
            mosaic[i*size:(i+1)*size, j*size:(j+1)*size] = im

    mosaic = Image.fromarray(mosaic)
    mosaic.save(f'{save_dir}/mosaic.tif')