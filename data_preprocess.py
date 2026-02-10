import numpy as np
import random
import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch


class RndRotateTransform:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    
    
class NanoCT_Pair_Dataset(Dataset):
    def __init__(self, configs, transform=True):
        """
        Dataset for NanoCT denoising using paired data.
        
        Parameters
        ----------
        configs : dict
            Configuration dictionary containing:
            - "train_size": Number of training data.
            - "img_size": Size to which images are resized.
            - "train_sample_data": Path to the directory containing sample images.
            - "train_ref_data": Path to the directory containing reference images.
        transform : bool, optional
            Whether to apply data augmentation transforms. Default is True.
        """
        self.train_size = configs["train_size"]
        self.size = configs["img_size"]
        self.transform = transform
        dref_files = sorted(glob.glob(f"{configs["train_sample_data"]}/*.tif"))
        ref_files = glob.glob(f"{configs["train_ref_data"]}/*.tif")

        # load sample and reference images
        sample_imgs, ref_imgs = [], []
        for f in tqdm(dref_files, desc="Loading sample images"):
            raw_sample = Image.open(f).resize((self.size, self.size))
            # if the image is 8 bits, devide by 255 to normalize to [0, 1]
            if raw_sample.mode == "L":
                raw_sample = np.array(raw_sample) / 255.0 
            else:
                raw_sample = np.array(raw_sample)
            sample_imgs.append(raw_sample)
        for f in tqdm(ref_files, desc="Loading reference images"):
            raw_ref = Image.open(f).convert("I").resize((self.size, self.size)) 
            raw_ref = np.array(raw_ref)
            ref_imgs.append(raw_ref/raw_ref.max())

        self.n_samples = len(sample_imgs)
        self.n_refs = len(ref_imgs)
        self.input_imgs = torch.from_numpy(np.array(sample_imgs)).unsqueeze(1).float() # (N, 1, H, W)
        self.target_imgs = torch.from_numpy(np.array(ref_imgs)).unsqueeze(1).float()   # (N, 1, H, W)

    def __getitem__(self, index):
        id_1, id_2 = random.randint(0, self.n_samples-1), random.randint(0, self.n_samples-1)
        x_1, x_2 = self.input_imgs[id_1], self.input_imgs[id_2]
        ref = self.target_imgs[random.randint(0, self.n_refs-1)]

        # Data augmentation
        if self.transform:
            dref_aug = transforms.Compose([
                RndRotateTransform(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(self.size, scale=(0.5, 1), ratio=(3/4, 4/3))
                                        ])
            
            ref_aug = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                                        ])
            x_1, x_2, ref = dref_aug(x_1), dref_aug(x_2), ref_aug(ref)
        x = torch.cat([x_1*ref, x_2*ref], axis=0)
        return x, ref

    def __len__(self):
        return self.train_size