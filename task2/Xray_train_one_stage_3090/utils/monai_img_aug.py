import os
from torch.utils import data
import pandas as pd
import monai
import monai.transforms as mt
import h5py
import torch
import numpy as np
from monai.utils import set_determinism
import math
import random


def data_preprocess_t(keys, ms_dict):
    preprocess_transform = mt.Compose([
        mt.EnsureChannelFirstd(keys=keys, channel_dim='no_channel'),
        mt.NormalizeIntensityd(keys='CT', subtrahend=ms_dict['ct_mean'], divisor=ms_dict['ct_std']),
        mt.NormalizeIntensityd(keys='PET', subtrahend=ms_dict['pet_mean'], divisor=ms_dict['pet_std']),
        mt.NormalizeIntensityd(keys='organ', subtrahend=0, divisor=11.0),
        mt.ToTensord(keys=keys)
    ])
    return preprocess_transform


def data_monai_aug(keys):
    """
    输入3维图像和标签（不够维数的需要在最后面补维）,返回进行了相同扩增的图像和标签,输入输出格式为numpy
    imgs_all的通道数代表了有多少模态；
    masks_all是所有标签类型的数据，其最后一个通道是GT
    """
    interpolate_mode = ["bilinear" if i == 'CT' or i == "PET" else "nearest" for i in keys]
    rand_aug_transform = mt.Compose([
        mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),  # 翻转
        mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),  # 翻转
        mt.RandZoomd(keys=keys, prob=0.5, min_zoom=1, max_zoom=1.2, mode=interpolate_mode),  # 随机放大
        mt.RandAffined(
            keys=keys,
            prob=0.5,  # 0.5概率触发
            rotate_range=math.pi / 4,  # 旋转±45°
            shear_range=math.pi / 12,  # 剪切变换15°
            translate_range=((-50, 50), (-50, 50)),  # 平移50个像素
            scale_range=((-0.1, 0.1), (-0.1, 0.1)),  # 缩放0.75-1.25之间
            mode=interpolate_mode,  # 插值模式
            padding_mode='reflection',  # 边缘填充
        ),
        mt.RandAdjustContrastd(keys=['CT', 'PET'], prob=0.5, gamma=(0.8, 2.5)),  # 随机调节对比度
        mt.Rand2DElasticd(
            keys=keys,
            spacing=30,
            prob=0.5,
            magnitude_range=(0, 2),
            mode=interpolate_mode,
        ),  # 弹性变换  需要确认参数
        mt.RandGaussianSharpend(keys=['CT', 'PET'], prob=0.5, alpha=(1, 5)),  # 锐化
    ])
    return rand_aug_transform


class Image2dMultimodH5(data.Dataset):
    def __init__(self, h5list, image_size=512, mode='train', augmentation_prob=0.4):
        self.h5_paths = h5list
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.norm = False  # 标准化的三种选择： False-用全身统计结果做z标准化； 'z-score'-z标准化； 'normal'-归一化；
        xlsx_path = r'/data/nas/heyixue_group/LCa/mean_std_320版本.xls'
        ms_list = pd.read_excel(xlsx_path, 'in')[1:]
        # mean std dict
        self.keys = ['CT', 'PET', 'organ', 'mask']
        self.aug_transform = data_monai_aug(keys=self.keys)
        self.ms_dict = dict(ct_mean=list(ms_list['ct_mean']),
                            ct_std=list(ms_list['ct_std']),
                            pet_mean=list(ms_list['pet_mean']),
                            pet_std=list(ms_list['pet_std']))
        if not self.norm:
            print('标准化方法：3D全身标准化')
        else:
            print('标准化方法：2D的' + self.norm)

    def __getitem__(self, index):
        h5_path = self.h5_paths[index]
        file_id = int(h5_path.split('/')[-1].split('_')[0])
        h5_data = h5py.File(h5_path, 'r')
        img_dict = {'CT': h5_data['CT'][()],
                    'PET': h5_data['PET'][()],
                    'organ': h5_data['organ'][()],
                    'mask': h5_data['mask'][()]}

        ms_instance_dict = dict(
            pet_mean=self.ms_dict['pet_mean'][file_id],
            pet_std=self.ms_dict['pet_std'][file_id],
            ct_mean=self.ms_dict['ct_mean'][file_id],
            ct_std=self.ms_dict['ct_std'][file_id]
        )

        data_preprocess = data_preprocess_t(self.keys, ms_instance_dict)
        img_dict = data_preprocess(img_dict)  # 加通道，Z标准化，器官归一化，tensor化
        p_transform = random.random()  # 是否扩增
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            # 扩增操作
            img_dict = self.aug_transform(img_dict)
        # print(np.unique(img_dict['mask'].numpy()))
        # if img_dict['organ'][:, :].max() > 0:  # 器官数据归一化 （除11而不是该图层的最大值？），同时全0的则不需要归一化
        #     label_temp = (img_dict['organ'] / 11.0).astype(torch.float32)
        # else:
        #     label_temp = img_dict['organ'].astype(torch.float32)
        # img_dict['organ'] = label_temp
        image = torch.cat([img_dict['CT'].astype(torch.float32),
                           img_dict['PET'].astype(torch.float32),
                           img_dict['organ'].astype(torch.float32)], dim=0)

        return h5_path, image, img_dict['mask'].astype(torch.uint8)

    def __len__(self):
        return len(self.h5_paths)


def dataloader_monai_trans(h5_list, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4,
                           shuffle=True):
    """Builds and returns Dataloader."""

    dataset = Image2dMultimodH5(h5list=h5_list, image_size=image_size, mode=mode,
                                augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
    return data_loader
