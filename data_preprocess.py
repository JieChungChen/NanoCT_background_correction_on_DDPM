import numpy as np
import random, glob
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
    
class RndScaler:
    def __init__(self, f):
        self.factor = f

    def __call__(self, x):
        factor = random.uniform(self.factor[0], self.factor[1])
        return x*factor
  

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
            raw_ref = Image.open(ref_files[i]).convert("I").resize((img_size, img_size)) # 每個pixel是光強度，遠超255
            ref_imgs.append(np.array(raw_ref))
        dref_imgs = torch.from_numpy(np.array(dref_imgs)).unsqueeze(1).float()
        ref_imgs = torch.from_numpy(np.array(ref_imgs)).unsqueeze(1).float()
        
        self.input_imgs = dref_imgs.repeat_interleave(100, dim=0)
        self.input_imgs = self.input_imgs
        self.target_imgs = ref_imgs
        self.target_imgs = self.target_imgs.repeat(num_sample, 1, 1, 1)

    def __getitem__(self, index):
        n_samples = len(self.input_imgs)
        x_1, x_2 = self.input_imgs[index], self.input_imgs[random.randint(0, n_samples-1)]
        ref = self.target_imgs[index]
        ref = ref/ref.max()
        if self.transform:
            dref_aug = transforms.Compose([
                RndRotateTransform(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(self.size, scale=(0.5, 1), ratio=(3/4, 4/3))
                                        ])
            
            ref_aug = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                RndScaler(f=(0.75, 1.33)),
                                        ])
            x_1, x_2, ref = dref_aug(x_1), dref_aug(x_2), ref_aug(ref)
        x = torch.cat([x_1*ref, x_2*ref], axis=0)
        return x, ref

    def __len__(self):
        return len(self.input_imgs)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = NanoCT_Pair_Dataset('./training_data_n', 256)
    for i in range(10):
        x, ref = data[i]
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