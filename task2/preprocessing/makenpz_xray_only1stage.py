import cv2
import numpy as np
import SimpleITK as sitk
import h5py
import os
import pengwin_utils


def extract_region_with_padding(image, mask, padding=5):

    # 将图像和mask转换为numpy数组
    image_array = image
    mask_array = mask

    # 找到mask中非零部分的bounding box
    non_zero_indices = np.argwhere(mask_array)
    # range_indices = np.argwhere((mask_array >= start_index) & (mask_array <= end_index))

    min_y, min_x = non_zero_indices.min(axis=0)
    max_y, max_x = non_zero_indices.max(axis=0)
    # print(min_y, min_x, max_y, max_x)
    # 扩展bounding box大小
    min_y = max(min_y - padding, 0)
    min_x = max(min_x - padding, 0)
    max_y = min(max_y + padding, mask_array.shape[0] - 1)
    max_x = min(max_x + padding, mask_array.shape[1] - 1)

    # 提取扩展后的区域
    extracted_image_array = image_array[min_y:max_y + 1, min_x:max_x + 1]
    extracted_mask_array = mask_array[min_y:max_y + 1, min_x:max_x + 1]
    roi_info = [min_y, min_x, max_y, max_x]

    return extracted_image_array, extracted_mask_array,roi_info

def get_indices(lst: list[int], value: int) -> list[int]:
    """
    获取列表中所有等于特定值的元素的下标。

    参数:
        lst (list[int]): 原始列表。
        value (int): 需要查找的特定值。

    返回:
        list[int]: 所有等于特定值的元素的下标。
    """
    return [index for index, val in enumerate(lst) if val == value]

def merge_fragments_by_category(masks: np.ndarray, category_ids: list[int], fragment_ids: list[int]) -> np.ndarray:
    """
    将同一大骨区域的碎片细分。先是骶骨，再左髋骨，最后右髋骨（我的视角）
    """
    h, w = masks.shape[1], masks.shape[2]
    mask_list = []
    # 遍历三大骨区域
    for i in range(NUM):
        # 每个大骨区域10个碎片
        merged_masks = np.zeros((10, h, w), dtype=np.int8)
        category_indices = get_indices(category_ids, i + 1)
        count = 0
        # 遍历三个类别
        for category_index in category_indices:
            mask = masks[category_index]
            mask[mask != 0] = 1  # 将掩码中的非零值设置为1
            merged_masks[count] = mask
            count += 1
        mask_list.append(merged_masks)
    return mask_list


xray_dir = r'/home/nas/pzh/PENGWIN/task2/train/input/images/x-ray'
mask_dir = r'/home/nas/pzh/PENGWIN/task2/train/output/images/x-ray'
# xray_dir = r'E:\documents\train\input\images\x-ray'
# mask_dir = r'E:\documents\train\output\images\x-ray'
# xray_dir = r'D:\PENGWIN\task2\train\input\images\x-ray'
# mask_dir = r'D:\PENGWIN\task2\train\output\images\x-ray'
# h5_save_folder = r'/home/siat/pzh/pengwin/two_stage/h5_data'
# h5_save_folder = r'/home/siat/pzh/pengwin/two_stage/h5_data'
# h5_save_folder = r'/home/medig/pzh/pengwin/task2/h5_data/'
npz_save_folder = r'/home/nas/pzh/PENGWIN/task2/h5_data_only1stage'

if not os.path.exists(npz_save_folder):
    os.makedirs(npz_save_folder)


image_list = os.listdir(xray_dir)
seg_list = os.listdir(mask_dir)
image_list = sorted(image_list)
seg_list = sorted(seg_list)
NUM = 3

for image, seg in zip(image_list, seg_list):
    id = seg.split('.')[0]
    print(id)
    image_path = os.path.join(xray_dir, image)
    seg_path = os.path.join(mask_dir, seg)

    image = pengwin_utils.load_image(image_path)  # 原始强度图像
    masks, category_ids, fragment_ids = pengwin_utils.load_masks(seg_path)

    # 打印掩码形状和类别、碎片 ID
    # print("掩码形状:", masks.shape)
    # print("类别 ID:", category_ids)
    # print("碎片 ID:", fragment_ids)
    
    # 将布尔型掩码转换为合并后的多通道格式
    mask_list = merge_fragments_by_category(masks, category_ids, fragment_ids)
    merged_mask1 = mask_list[0]
    merged_mask2 = mask_list[1]
    merged_mask3 = mask_list[2]
    # print("合并后的掩码形状:", merged_mask2.shape)

    # # 可视化提取后的区域
    # extracted_nii1 = sitk.GetImageFromArray(image)
    # sitk.WriteImage(extracted_nii1, "extracted_xray2.nii.gz")
    # extracted_nii3 = sitk.GetImageFromArray(merged_mask1[0])
    # sitk.WriteImage(extracted_nii3, "extracted_xray2_0c.nii.gz")
    # extracted_nii3 = sitk.GetImageFromArray(merged_mask1[1])
    # sitk.WriteImage(extracted_nii3, "extracted_xray2_1c.nii.gz")

    # 保存为h5
    sep = os.sep
    npz_save_file = npz_save_folder + sep + id + '.npz'
    # print(npz_save_file)

    # 实际保存的是原图X光加三个10通道的mask


    # 保存到一个压缩的 .npz 文件中
    np.savez_compressed(npz_save_file, 
                        xray=image, 
                        sacrum_mask=merged_mask1, 
                        hipleft_mask=merged_mask2, 
                        hipright_mask=merged_mask3)

    # break







