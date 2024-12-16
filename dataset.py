import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import os


class Dataset3D(Dataset):
    def __init__(self, data_path, transforms, df=None):
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_excel(data_path)

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __fill_data__(self, data):
        # 计算需要填充的维度大小
        desired_d = max(data.shape[1], data.shape[2])
        desired_shape = (86, desired_d, desired_d)
        original_array = np.array(data)

        # 创建一个新的数组，全为0，大小为 desired_shape
        new_array = np.zeros(desired_shape, dtype=data.dtype)

        # 将原始数组的数据填充到新数组中，保证尺寸匹配
        new_array[:original_array.shape[0], :original_array.shape[1], :original_array.shape[2]] = original_array
        return new_array

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        id = item['id']
        label = item['label']
        path = item['path']

        data_t1c = sitk.ReadImage(os.path.join(path, 't1c', 't1c_cut.nii'))
        sample_t1c = sitk.GetArrayFromImage(data_t1c)
        sample_t1c = self.__fill_data__(sample_t1c)
        new_sample_t1c = []

        for s in sample_t1c:
            if self.transforms is not None:
                s = self.transforms(Image.fromarray(s)).unsqueeze(0)
            new_sample_t1c.append(s)
        new_sample_t1c = torch.cat(new_sample_t1c, dim=0)
        new_sample_t1c = new_sample_t1c.permute(1, 0, 2, 3)


        data_t2h = sitk.ReadImage(os.path.join(path, 't2h', 't2h_cut.nii'))
        sample_t2h = sitk.GetArrayFromImage(data_t2h)
        sample_t2h = self.__fill_data__(sample_t2h)
        new_sample_t2h = []
        for s in sample_t2h:
            if self.transforms is not None:
                s = self.transforms(Image.fromarray(s).convert('L')).unsqueeze(0)
            new_sample_t2h.append(s)
        new_sample_t2h = torch.cat(new_sample_t2h, dim=0)
        new_sample_t2h = new_sample_t2h.permute(1, 0, 2, 3)

        combined_sample = torch.cat([new_sample_t1c, new_sample_t2h], dim=0)


        return combined_sample, label













