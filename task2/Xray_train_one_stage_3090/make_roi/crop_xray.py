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
h5_save_folder = r'/home/siat/pzh/pengwin/two_stage/h5_data'


h5_hip_left_dir = os.path.join(h5_save_folder, 'hip_left')
h5_hip_right_dir = os.path.join(h5_save_folder, 'hip_right')
h5_sacrum_dir = os.path.join(h5_save_folder, 'sacrum')

if not os.path.exists(h5_save_folder):
    os.makedirs(h5_save_folder)
    os.makedirs(h5_hip_left_dir)
    os.makedirs(h5_hip_right_dir)
    os.makedirs(h5_sacrum_dir)


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
    print("掩码形状:", masks.shape)
    print("类别 ID:", category_ids)
    print("碎片 ID:", fragment_ids)
    # 将布尔型掩码转换为合并后的多通道格式
    mask_list = merge_fragments_by_category(masks, category_ids, fragment_ids)
    merged_mask1 = mask_list[0]
    merged_mask2 = mask_list[1]
    merged_mask3 = mask_list[2]

    print("合并后的掩码形状:", merged_mask2.shape)
    # print(np.unique(merged_mask1))
    # print(np.unique(merged_mask2))
    # print(np.unique(merged_mask3))

    merged_mask1_temp = np.zeros((merged_mask1.shape[1], merged_mask1.shape[2]))
    merged_mask2_temp = np.zeros((merged_mask2.shape[1], merged_mask2.shape[2]))
    merged_mask3_temp = np.zeros((merged_mask3.shape[1], merged_mask3.shape[2]))

    # 提取感兴趣区域
    # 先把10通道区域取并集
    for i in range(merged_mask1.shape[0]):
        merged_mask1_temp += merged_mask1[i,:,:] 
    merged_mask1_temp[merged_mask1_temp!=0] = 1

    for i in range(merged_mask2.shape[0]):
        merged_mask2_temp += merged_mask2[i,:,:] 
    merged_mask2_temp[merged_mask2_temp!=0] = 1

    for i in range(merged_mask3.shape[0]):
        merged_mask3_temp += merged_mask3[i,:,:] 
    merged_mask3_temp[merged_mask3_temp!=0] = 1

    # 提取感兴趣区域

    if np.sum(merged_mask1_temp) != 0:
        extracted_xray1, extracted_mask1, roi_info1 = extract_region_with_padding(image, merged_mask1_temp, padding=5)
    if np.sum(merged_mask2_temp) != 0:
        extracted_xray2, extracted_mask2, roi_info2 = extract_region_with_padding(image, merged_mask2_temp, padding=5)
    if np.sum(merged_mask3_temp) != 0:
        extracted_xray3, extracted_mask3, roi_info3 = extract_region_with_padding(image, merged_mask3_temp, padding=5)

    # 可视化提取后的区域
    # extracted_nii1 = sitk.GetImageFromArray(extracted_xray1)
    # sitk.WriteImage(extracted_nii1, "extracted_xray1.nii.gz")
    # extracted_nii2 = sitk.GetImageFromArray(extracted_xray2)
    # sitk.WriteImage(extracted_nii2, "extracted_xray2.nii.gz")
    # extracted_nii3 = sitk.GetImageFromArray(extracted_xray3)
    # sitk.WriteImage(extracted_nii3, "extracted_xray3.nii.gz")
    
    # extracted_nii3 = sitk.GetImageFromArray(merged_mask2[0:1,roi_info2[0]: roi_info2[2]+1, roi_info2[1]: roi_info2[3]+1])
    # sitk.WriteImage(extracted_nii3, "extracted_xray2_0c.nii.gz")
    # extracted_nii3 = sitk.GetImageFromArray(merged_mask2[1:2,roi_info2[0]: roi_info2[2]+1, roi_info2[1]: roi_info2[3]+1])
    # sitk.WriteImage(extracted_nii3, "extracted_xray2_1c.nii.gz")
    
    # extracted_xray_return = image[roi_info3[0]:roi_info3[2], roi_info3[1]:roi_info3[3]]
    # extracted_nii3 = sitk.GetImageFromArray(extracted_xray3)
    # sitk.WriteImage(extracted_nii3, "extracted_xray3_return.nii.gz")
    
    # extracted_nii1 = sitk.GetImageFromArray(extracted_mask1)
    # sitk.WriteImage(extracted_nii1, "extracted_mask1.nii.gz")
    # extracted_nii2 = sitk.GetImageFromArray(extracted_mask2)
    # sitk.WriteImage(extracted_nii2, "extracted_mask2.nii.gz")
    # extracted_nii3 = sitk.GetImageFromArray(extracted_mask3)
    # sitk.WriteImage(extracted_nii3, "extracted_mask3.nii.gz")

    # 保存为h5
    sep = os.sep
    hip_left_save_file = h5_hip_left_dir + sep + id + '.h5'
    hip_right_save_file = h5_hip_right_dir + sep + id + '.h5'
    sacrum_file = h5_sacrum_dir + sep + id + '.h5'

    print(hip_left_save_file)
    # 实际保存的是骶骨
    # h5_file = h5py.File(sacrum_file, 'w')
    # h5_file['xray'] = extracted_xray1
    # h5_file['mask'] = merged_mask1[:,roi_info1[0]: roi_info1[2]+1, roi_info1[1]: roi_info1[3]+1]
    # h5_file['crop_info'] = roi_info1
    # h5_file.close()

    # 实际保存的是左髋骨
    h5_file = h5py.File(hip_left_save_file, 'w')
    h5_file['xray'] = extracted_xray2
    h5_file['mask'] = merged_mask2[:,roi_info2[0]: roi_info2[2]+1, roi_info2[1]: roi_info2[3]+1]
    h5_file['crop_info'] = roi_info2
    h5_file.close()

    # 实际保存的是右髋骨
    # h5_file = h5py.File(hip_right_save_file, 'w')
    # h5_file['xray'] = extracted_xray3
    # h5_file['mask'] = merged_mask3[:,roi_info3[0]: roi_info3[2]+1, roi_info3[1]: roi_info3[3]+1]
    # # print(extracted_xray3.shape)
    # # print(merged_mask3[:,roi_info3[0]: roi_info3[2]+1, roi_info3[1]: roi_info3[3]+1].shape)
    # h5_file['crop_info'] = roi_info3
    # h5_file.close()


    # break







