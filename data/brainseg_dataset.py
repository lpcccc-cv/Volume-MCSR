import numpy as np
import torch
import torch.utils.data as data
import os
import random
import SimpleITK as sitk

def norm(data):
    data = data.astype(np.float32)
    # data = np.clip(data, a_min=-200, a_max=400)
    max = np.max(data)
    min = np.min(data)
    data = (data-min)/(max-min)
    return data*255.


class brainseg_train(data.Dataset):
    def __init__(self, opt, train):
        super(brainseg_train, self).__init__()
        
        if train == True:
            self.path = opt['train_path']
            if 'IXI' in self.path:
                ref_path = self.path.replace('T2', 'PD')
            else:
                ref_path = self.path.replace('T2', 'T1')
        else:
            self.path = opt['test_path']
            if 'IXI' in self.path:
                ref_path = self.path.replace('T2', 'PD')
            else:
                ref_path = self.path.replace('T2', 'T1')
        GT_volume_list = sorted(os.listdir(self.path))
        ref_volume_list = sorted(os.listdir(ref_path))

        # preprocess volume data
        self.all_GT_volume_data = []
        for i,volume_name in enumerate(GT_volume_list):
            if i>3:
                continue
            GT_volume_path = os.path.join(self.path,volume_name)
            volume_data = sitk.ReadImage(GT_volume_path, sitk.sitkInt16)
            volume_data = sitk.GetArrayFromImage(volume_data)
            self.all_GT_volume_data.append(norm(volume_data))
            print('Pre-precess GT:', volume_name, volume_data.shape)

        self.all_ref_volume_data = []
        for j, volume_name in enumerate(ref_volume_list):
            if j>3:
                continue
            ref_volume_path = os.path.join(ref_path,volume_name)
            volume_data = sitk.ReadImage(ref_volume_path, sitk.sitkInt16)
            volume_data = sitk.GetArrayFromImage(volume_data)
            self.all_ref_volume_data.append(norm(volume_data))
            print('Pre-precess Ref:', volume_name, volume_data.shape)
        
        self.train = train
        self.crop_size = opt['crop_size']
        self.scale = int(opt['scale'])

        if opt['task'] == 'sr_volume':
            self.d_size = opt['d_size']
        else:
            print('######## Error! #########')


    def __len__(self):

        return len(self.all_GT_volume_data)


    def __getitem__(self, idx):

        # read image file
        volume_GT = self.all_GT_volume_data[idx]
        volume_GT = torch.tensor(volume_GT).unsqueeze(1).float()/255.

        volume_ref = self.all_ref_volume_data[idx]
        volume_ref = torch.tensor(volume_ref).unsqueeze(1).float()/255.


        # 随机裁剪
        if self.train:
            D, _, H, W = volume_GT.shape 
            rnd_h = random.randint(0, max(0, H - self.crop_size))
            rnd_w = random.randint(0, max(0, W - self.crop_size))
            rnd_d = random.randint(0, max(0, D - self.d_size))
            volume_GT = volume_GT[rnd_d:rnd_d+self.d_size, :, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
            volume_ref = volume_ref[rnd_d:rnd_d+self.d_size, :, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]

        else:
            D, _, H, W = volume_GT.shape 
            rnd_h = 100
            rnd_w = 100
            if 'IXI' in self.path:
                rnd_d = 50
            else:
                rnd_d = 100
            volume_GT = volume_GT[rnd_d:rnd_d+self.d_size]
            volume_ref = volume_ref[rnd_d:rnd_d+self.d_size]

        # 欠采样
        volume_LQ = volume_GT[::self.scale]

        # print(volume_LQ.shape, volume_GT.shape, volume_ref.shape)

        return {'volume_LQ':volume_LQ, 'volume_GT':volume_GT, 'volume_ref':volume_ref}
        
