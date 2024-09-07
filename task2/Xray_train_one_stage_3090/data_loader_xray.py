# import sys
# sys.path.append('/home/nas/pzh/code/MyModel_Seg_A100/seg_code_2d')
import random
import torch
from torch.utils import data
from torchvision import transforms as T
from utils.img_mask_aug import *
import h5py
import numpy as np
import scipy.ndimage

# nnunet文件统计出来的数值
# 0.05和99.5分位值
# ----------------hip_left-----------------
# "percentile_00_5": 0.03138490378856659,
# "percentile_99_5": 3.5369632291793778,
# "mean": 0.6797539846280519,
# "std": 0.6073874151137711

# ----------------hip_right-----------------
# percentile_00_5 = 0.03209742857143283
# percentile_99_5 = 4.22320652246475
# "mean": 0.7187395914616901,
# "std": 0.7253404895759054
# ----------------sacrum-----------------

# "percentile_00_5": 0.025976749137043953,
# "percentile_99_5": 2.2021718049049355,
# "mean": 0.550791488199508,
# "std": 0.41074393110036345


# def hu_clipping_and_normalization(data, percentile_00_5=0.03209742857143283, percentile_99_5=4.22320652246475,
#                                   mean=0.0, std=1.0):
#     # Step 1: HU Clipping
#     clipped_data = np.clip(data, percentile_00_5, percentile_99_5)
    
#     # Step 2: Normalization 
#     normalized_data = (clipped_data - mean) / max(std, 1e-8)

#     return normalized_data

def zscore_normalize_image(image):
    """
    Normalize the image to have mean 0 and variance 1.

    Parameters:
    image (np.ndarray): Input image.

    Returns:
    np.ndarray: Normalized image.
    """
    # Calculate mean and standard deviation
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= (max(std, 1e-8))
    # Normalize the image
    return image

class ImageFolder_2d_xray_npz(data.Dataset):
    def __init__(self, npz_list, image_size=448, mode='train', augmentation_prob=0.4, dataset='hip_left'):
        """Initializes image paths and preprocessing module."""
        # self.root = root

        # GT : Ground Truth
        # self.GT_paths = os.path.join(root, 'p_mask')
        self.npz_paths = npz_list
        self.dataset = dataset
        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.norm = 'z-score'   # 标准化的三种选择： False-用全身统计结果做z标准化 'z-score'-z标准化 'normal'-归一化

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        npz_path = self.npz_paths[index]
        npz_data = np.load(npz_path)

        # print(f'filename{filename}')
        # 访问并读取各个数组
        image = npz_data['xray']
        hipright_mask = npz_data['hipright_mask']

        xray = image.astype(np.float32) #(448,448)
        
        GT = hipright_mask                #(10,448,448)
        GT = GT / 1.0

        # 归一化
        xray = zscore_normalize_image(xray)
        GT = np.transpose(GT, (1, 2, 0))

        # print(GT.shape)
        image = xray[:, :, np.newaxis]


        p_transform = random.random()  # 是否扩增
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            # 扩增操作
            [image, GT] = data_aug_multimod(image, GT)     # 扩增操作

        # 确保大小正确+tensor化
        Transform = []
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        image = image.type(torch.FloatTensor)
        GT = Transform(GT)
        # return h5_path, image, GT ,filename
        return npz_path, image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.npz_paths)


def get_loader_2d_xray_npz(h5_list, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4,dataset='hip_left',
                           shuffle=True):
    """Builds and returns Dataloader."""
    dataset = ImageFolder_2d_xray_npz(npz_list=h5_list, image_size=image_size, mode=mode,
                                      augmentation_prob=augmentation_prob)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
    return data_loader

# class ImageFolder_2d_xray_h5(data.Dataset):
#     def __init__(self, h5list, image_size=448, mode='train', augmentation_prob=0.4, dataset='hip_left'):
#         """Initializes image paths and preprocessing module."""
#         # self.root = root

#         # GT : Ground Truth
#         # self.GT_paths = os.path.join(root, 'p_mask')
#         self.h5_paths = h5list
#         self.dataset = dataset
#         self.image_size = image_size
#         self.mode = mode
#         self.augmentation_prob = augmentation_prob
#         self.norm = 'z-score'   # 标准化的三种选择： False-用全身统计结果做z标准化 'z-score'-z标准化 'normal'-归一化

#     def __getitem__(self, index):
#         """Reads an image from a file and preprocesses it and returns."""
#         h5_path = self.h5_paths[index]
#         filename = h5_path.split('/')[-1]
#         # print(f'filename{filename}')
#         h5_data = h5py.File(h5_path, 'r')

#         xray = h5_data['xray'][()].astype(np.float32) #(448,448)
#         GT = h5_data['mask'][()]                #(10,448,448)
#         GT = GT / 1.0
#         # 获取原始图像的大小
#         original_size = xray.shape
#         # 使用双线性插值放大图像到448x448
#         xray = scipy.ndimage.zoom(xray, (self.image_size / original_size[0], self.image_size / original_size[1]), order=1)

#         # 对每个通道的标签使用最近邻插值放大到10，448，448
#         resized_mask_data = np.zeros((GT.shape[0], self.image_size, self.image_size))
#         for i in range(GT.shape[0]):
#             resized_channel = scipy.ndimage.zoom(GT[i, :, :],
#                                                  (self.image_size / original_size[0], self.image_size / original_size[1]),
#                                                  order=0)
#             resized_mask_data[i, :, :] = resized_channel

#         GT = resized_mask_data
#         del resized_mask_data
#         # 归一化
#         xray = zscore_normalize_image(xray)

#         GT = np.transpose(GT, (1, 2, 0))

#         # print(GT.shape)
#         image = xray[:, :, np.newaxis]


#         p_transform = random.random()  # 是否扩增
#         if (self.mode == 'train') and p_transform <= self.augmentation_prob:
#             # 扩增操作
#             [image, GT] = data_aug_multimod(image, GT)     # 扩增操作

#         # 确保大小正确+tensor化
#         Transform = []
#         Transform.append(T.ToTensor())
#         Transform = T.Compose(Transform)

#         image = Transform(image)
#         image = image.type(torch.FloatTensor)
#         GT = Transform(GT)
#         # return h5_path, image, GT ,filename
#         return h5_path, image, GT

#     def __len__(self):
#         """Returns the total number of font files."""
#         return len(self.h5_paths)


# def get_loader_2d_xray_h5(h5_list, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4,dataset='hip_left',
#                            shuffle=True):
#     """Builds and returns Dataloader."""
#     dataset = ImageFolder_2d_xray_h5(h5list=h5_list, image_size=image_size, mode=mode,
#                                       augmentation_prob=augmentation_prob)

#     data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
#                                   shuffle=shuffle, num_workers=num_workers,
#                                   pin_memory=True, drop_last=True)
#     return data_loader
