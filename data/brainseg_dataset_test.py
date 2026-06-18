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


class brainseg_test(data.Dataset):
    def __init__(self, scale, test_path):
        super(brainseg_test, self).__init__()
        
        self.path = test_path
        if 'IXI' in self.path:
            ref_path = self.path.replace('T2', 'PD')
        else:
            ref_path = self.path.replace('T2', 'T1')
        GT_volume_list = sorted(os.listdir(self.path))
        ref_volume_list = sorted(os.listdir(ref_path))

        # preprocess volume data
        self.all_GT_volume_data = []
        for volume_name in GT_volume_list:
            GT_volume_path = os.path.join(self.path,volume_name)
            volume_data = sitk.ReadImage(GT_volume_path, sitk.sitkInt16)
            volume_data = sitk.GetArrayFromImage(volume_data)
            self.all_GT_volume_data.append(norm(volume_data))
            print('Pre-precess GT:', volume_name, volume_data.shape)

        self.all_ref_volume_data = []
        for volume_name in ref_volume_list:
            ref_volume_path = os.path.join(ref_path,volume_name)
            volume_data = sitk.ReadImage(ref_volume_path, sitk.sitkInt16)
            volume_data = sitk.GetArrayFromImage(volume_data)
            self.all_ref_volume_data.append(norm(volume_data))
            print('Pre-precess Ref:', volume_name, volume_data.shape)
        
        self.scale = scale


    def __len__(self):

        return len(self.all_GT_volume_data)

    def __getitem__(self, idx):

        # read image file
        volume_GT = self.all_GT_volume_data[idx]
        volume_GT = torch.tensor(volume_GT).unsqueeze(1).float()/255.

        volume_ref = self.all_ref_volume_data[idx]
        volume_ref = torch.tensor(volume_ref).unsqueeze(1).float()/255.

        # 欠采样
        volume_LQ = volume_GT[::self.scale]

        return {'volume_LQ':volume_LQ, 'volume_GT':volume_GT, 'volume_ref':volume_ref}