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
            raw_ref = Image.open(ref_files[i]).resize((img_size, img_size)) # 每個pixel是光強度，遠超255
            ref_imgs.append(np.array(raw_ref))
        dref_imgs = torch.from_numpy(np.array(dref_imgs)).unsqueeze(1)
        ref_imgs = torch.from_numpy(np.array(ref_imgs)).unsqueeze(1)
        
        self.input_imgs = [dref*ref for dref in dref_imgs for ref in ref_imgs]
        self.input_imgs = torch.cat(self.input_imgs, axis=0).unsqueeze(1)
        # normalize input images to [0, 1]
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
                RndRotateTransform(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(self.size, scale=(0.9, 1), ratio=(1, 1))
            ])
            transformed = aug(torch.cat([x, ref], dim=0))
            x, ref = transformed[:1], transformed[1:]
        return x, ref

    def __len__(self):
        return len(self.input_imgs)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = NanoCT_Dataset('./training_data_n', 128)
    for i in range(10):
        x, ref = data[i]
        print(x.max(), ref.max())
        plt.subplot(121)
        plt.imshow(x.squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(ref.squeeze(), cmap='gray')
        plt.axis('off')
        plt.show()
        plt.close()