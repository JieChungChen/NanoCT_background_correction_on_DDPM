import os, glob
import torch
import numpy as np
import matplotlib as mpl
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