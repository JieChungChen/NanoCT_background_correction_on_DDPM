import numpy as np
import glob
import random
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import time


class RndRotateTransform:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    

class NanoCT_Dataset(Dataset):
    def __init__(self, data_dir, img_size, num_sample=100, transform=True):
        t_start = time.time()
        self.size = img_size
        self.transform = transform
        dref_files = sorted(glob.glob("%s/dref/*.tif"%data_dir))
        ref_files = sorted(glob.glob("%s/ref/*.tif"%data_dir))

        dref_imgs, ref_imgs = [], []
        dref_rnd_choose = np.random.choice(len(dref_files), num_sample, replace=False)
        ref_rnd_choose = np.random.choice(len(ref_files), num_sample, replace=False)
        for i in tqdm(dref_rnd_choose, dynamic_ncols=True, desc='load dref images'):
            raw_dref = Image.open(dref_files[i]).resize((img_size, img_size)) # 每個pixel是光強度，遠超255
            dref_imgs.append(np.array(raw_dref))
        for i in tqdm(ref_rnd_choose, dynamic_ncols=True, desc='load ref images'):
            raw_ref = np.array(Image.open(ref_files[i]).resize((img_size, img_size))) # 每個pixel是光強度，遠超255
            ref_imgs.append(raw_ref)
        dref_imgs = torch.from_numpy(np.array(dref_imgs)).unsqueeze(1).float()
        ref_imgs = torch.from_numpy(np.array(ref_imgs)).unsqueeze(1).float()
        
        self.input_imgs = [dref*ref for dref in dref_imgs for ref in ref_imgs]
        self.input_imgs = torch.cat(self.input_imgs, axis=0).unsqueeze(1)
        inputs_max = self.input_imgs.amax(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        refs_max = ref_imgs.amax(dim=(1, 2, 3)).view(-1, 1, 1, 1)

        self.input_imgs = self.input_imgs/inputs_max
        self.target_imgs = ref_imgs/refs_max
        self.target_imgs = self.target_imgs.repeat(num_sample, 1, 1, 1)
        print('training data preprocessing finished: %.2f sec'%(time.time()-t_start))

    def __getitem__(self, index):
        x, ref = self.input_imgs[index], self.target_imgs[index]
        if self.transform:
            aug = transforms.Compose([
                # RndRotateTransform(),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomResizedCrop(self.size, scale=(0.9, 1), ratio=(1, 1))
            ])
            transformed = aug(torch.cat([x, ref], dim=0))
            x, ref = transformed[:1], transformed[1:]
        return x, ref

    def __len__(self):
        return len(self.input_imgs)
    

class NanoCT_Pair_Dataset(Dataset):
    def __init__(self, data_dir, img_size, num_sample=100, transform=True):
        self.size = img_size
        self.transform = transform
        dref_files = sorted(glob.glob("%s/dref/*.tif"%data_dir))
        ref_files = sorted(glob.glob("%s/ref/*.tif"%data_dir))

        dref_imgs, ref_imgs = [], []
        dref_rnd_choose = np.random.choice(len(dref_files), num_sample, replace=False)
        ref_rnd_choose = np.random.choice(len(ref_files), num_sample, replace=False)
        for i in dref_rnd_choose:
            raw_dref = Image.open(dref_files[i]).resize((img_size, img_size)) # 每個pixel是光強度，遠超255
            dref_imgs.append(np.array(raw_dref))
        for i in ref_rnd_choose:
            raw_ref = np.array(Image.open(ref_files[i]).resize((img_size, img_size))) # 每個pixel是光強度，遠超255
            ref_imgs.append(raw_ref)
        dref_imgs = torch.from_numpy(np.array(dref_imgs)).unsqueeze(1).float()
        ref_imgs = torch.from_numpy(np.array(ref_imgs)).unsqueeze(1).float()
        
        self.input_imgs = dref_imgs.repeat_interleave(100, dim=0)
        inputs_max = self.input_imgs.amax(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        refs_max = ref_imgs.amax(dim=(1, 2, 3)).view(-1, 1, 1, 1)

        self.input_imgs = self.input_imgs/inputs_max
        self.target_imgs = ref_imgs/refs_max
        self.target_imgs = self.target_imgs.repeat(num_sample, 1, 1, 1)

    def __getitem__(self, index):
        n_samples = len(self.input_imgs)
        x_1, x_2 = self.input_imgs[index], self.input_imgs[random.randint(0, n_samples-1)]
        ref = self.target_imgs[index]
        if self.transform:
            dref_aug = transforms.Compose([
                RndRotateTransform(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(self.size, scale=(0.5, 1), ratio=(3/4, 4/3))])
            
            ref_aug = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
            ])
            x_1, x_2, ref = dref_aug(x_1), dref_aug(x_2), ref_aug(ref)
        x = torch.cat([x_1*ref, x_2*ref], axis=0)
        return x, ref

    def __len__(self):
        return len(self.input_imgs)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = NanoCT_Pair_Dataset('./training_data_n', 256)
    for i in range(5):
        x, ref = data[i]
        print(x.shape, ref.shape)
        print(x.max(), ref.max())
        plt.subplot(131)
        plt.imshow(x[0].squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(x[1].squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(ref.squeeze(), cmap='gray')
        plt.axis('off')
        plt.show()
        plt.close()