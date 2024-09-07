from seg_code_2d.utils.monai_img_aug import data_monai_aug, data_preprocess_t
from seg_code_2d.utils.img_mask_aug import data_aug_multimod
import h5py
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from monai.utils import set_determinism
import os


def test_iaa(h5_data, ms_dict):
    image_all = None
    label_all = None
    GT_idx = 0
    for key in h5_data.keys():
        if key == 'mask' or key == 'organ':  # label输入，设定GT是h5的最后一个
            label = h5_data[key][()].astype(np.uint8)
            label = label[:, :, np.newaxis]
            if label_all is None:  # 最开始是没有label_all
                label_all = label
                if key == 'mask':  # 判断GT是在label_all的哪一层
                    GT_o = h5_data[key][()].astype(np.uint8)
                    GT_idx = 0
            else:
                label_all = np.concatenate((label_all, label), axis=2)
                if key == 'mask':  # 判断GT是在label_all的哪一层
                    GT_idx = label_all.shape[2] - 1
        else:
            img = h5_data[key][()].astype(np.float32)  # 图像输入
            if key == 'PET':
                img = (img - ms_dict['pet_mean']) / ms_dict['pet_std']
            else:  # key == 'CT'
                img = (img - ms_dict['ct_mean']) / ms_dict['ct_std']
            # 拼在一起，有多少个模态就多少通道
            img = img[:, :, np.newaxis]
            if image_all is None:  # 最开始是没有image_all
                image_all = img
            else:
                image_all = np.concatenate((image_all, img), axis=2)
    p_transform = random.random()  # 是否扩增
    if p_transform <= 0.5:
        # 扩增操作
        [image_all, label_all] = data_aug_multimod(image_all, label_all)
    return image_all, label_all, GT_idx


if __name__ == '__main__':
    set_determinism(seed=2022)
    random.seed(2022)
    os.environ["PYTHONASHSEED"] = str(2022)
    np.random.seed(seed=2022)
    path = r'/data/nas3/MJY_file/LCa/h5_data_pre/320'
    path = r'/data/nas/heyixue_group/LCa/h5_data_pre/320'
    h5_path = [path + '/418_' + str(i) + '.h5' for i in range(155, 165)]
    keys = ['CT', 'PET', 'organ', 'mask']
    aug_transform = data_monai_aug(keys=keys)
    for p in h5_path:
        h5_data = h5py.File(p, 'r')

        img_dict = {'CT': h5_data['CT'][()],
                    'PET': h5_data['PET'][()],
                    'organ': h5_data['organ'][()],
                    'mask': h5_data['mask'][()]}
        xlsx_path = r'/data/nas/heyixue_group/LCa/mean_std_320版本.xls'
        ms_list = pd.read_excel(xlsx_path, 'in')[1:]
        # mean std dict
        ms_dict = dict(
            ct_mean=list(ms_list['ct_mean']),
            ct_std=list(ms_list['ct_std']),
            pet_mean=list(ms_list['pet_mean']),
            pet_std=list(ms_list['pet_std'])
        )

        ms_instance_dict = dict(
            pet_mean=ms_dict['pet_mean'][418],
            pet_std=ms_dict['pet_std'][418],
            ct_mean=ms_dict['ct_mean'][418],
            ct_std=ms_dict['ct_std'][418]
        )

        # image_all, label_all, GT_idx = test_iaa(h5_data, ms_instance_dict)
        # ct = image_all[:, :, 0]
        # pet = image_all[:, :, 1]
        # organ = label_all[:, :, GT_idx - 1]
        # mask = label_all[:, :, GT_idx]

        data_preprocess = data_preprocess_t(keys, ms_instance_dict)
        img_dict = data_preprocess(img_dict)  # 加通道，Z标准化，器官归一化，tensor化
        p_transform = random.random()  # 是否扩增
        if p_transform <= 0.5:
            # 扩增操作
            img_dict = aug_transform(img_dict)
        ct = img_dict['CT'][0].numpy()
        pet = img_dict['PET'][0].numpy()
        organ = img_dict['organ'][0].numpy()
        mask = img_dict['mask'][0].numpy()

        # f, ax = plt.subplots(1, 1)
        # ax.imshow(image)
        plt.subplot(1, 4, 1)
        plt.imshow(ct)
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(pet)
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(organ)
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(mask)
        plt.axis('off')
        plt.show()
