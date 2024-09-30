import numpy as np
import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import time

class NanoCT_Dataset(Dataset):
    def __init__(self, data_dir, img_size, num_sample=100):
        t_start = time.time()
        dref_files = sorted(glob.glob("%s/dref/*.tif"%data_dir))
        ref_files = sorted(glob.glob("%s/ref/*.tif"%data_dir))

        dref_imgs, ref_imgs = [], []
        dref_rnd_choose = np.random.choice(len(dref_files), num_sample, replace=False)
        ref_rnd_choose = np.random.choice(len(ref_files), num_sample, replace=False)
        for i in tqdm(dref_rnd_choose, dynamic_ncols=True, desc='load dref images'):
            raw_dref = Image.open(dref_files[i]) # 每個pixel是光強度，遠超255
            dref_imgs.append(np.array(raw_dref))
        for i in tqdm(ref_rnd_choose, dynamic_ncols=True, desc='load ref images'):
            raw_ref = Image.open(ref_files[i]) # 每個pixel是光強度，遠超255
            ref_imgs.append(np.array(raw_ref))
        dref_imgs = torch.Tensor(np.array(dref_imgs)).unsqueeze(1)
        ref_imgs = torch.Tensor(np.array(ref_imgs)).unsqueeze(1)
        resize = transforms.Resize((img_size, img_size))
        dref_imgs, ref_imgs = resize(dref_imgs), resize(ref_imgs)
        # print(dref_imgs.shape(), ref_imgs.shape())

        self.input_imgs = [dref*ref for dref in dref_imgs for ref in ref_imgs]
        self.input_imgs = torch.concatenate(self.input_imgs, dim=0).unsqueeze(1)
        self.input_imgs = self.input_imgs/self.input_imgs.max()
        self.target_imgs = ref_imgs/ref_imgs.max()
        self.target_imgs = self.target_imgs.repeat(100, 1, 1, 1)
        # print(self.input_imgs.shape, self.target_imgs.shape)
        print('training data preprocessing finished: %.2f sec'%(time.time()-t_start))

    def __getitem__(self, index):
        x, ref = self.input_imgs[index], self.target_imgs[index]
        return x, ref

    def __len__(self):
        return len(self.input_imgs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = NanoCT_Dataset('./training_data_n', 128)
    for i in np.arange(100):
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(data[i][0].squeeze().numpy(), cmap='gray')
        plt.subplot(122)
        plt.imshow(data[i][1].squeeze().numpy(), cmap='gray')
        plt.show()
        plt.close()