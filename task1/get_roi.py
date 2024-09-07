import SimpleITK as sitk
import numpy as np
import os





def extract_region_with_padding(image_path, mask_path, output_image_path, output_mask_path, padding=5,start_index = 1, end_index = 10):
    
    # 读取图像和mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # 将图像和mask转换为numpy数组
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    # print(image_array.shape)
    # print(mask_array.shape)
    # 因为二阶段是单独推理，在取ROI的时候将另外两大骨区域的CT值置为背景可以防止网络误判骨折碎片
    # background_value = np.min(image_array) - 1
    # if start_index == 1 and end_index == 10:
    #     # print('start_index == 1 and end_index == 10')
    #     NonTargetRegion = (mask_array >= 10) & (mask_array <= 30)
    #     image_array[NonTargetRegion] = background_value
    #     mask_array[NonTargetRegion] = 0

    # elif start_index == 11 and end_index == 20:
    #     # print('start_index == 11 and end_index == 20')
    #     NonTargetRegion = ((mask_array >= 1) & (mask_array <= 10)) | ((mask_array >= 21) & (mask_array <= 30))
    #     image_array[NonTargetRegion] = background_value
    #     mask_array[NonTargetRegion] = 0

    # elif start_index == 21 and end_index == 30:
    #     # print('start_index == 21 and end_index == 30')
    #     NonTargetRegion = (mask_array >= 1) & (mask_array <= 20)
    #     image_array[NonTargetRegion] = background_value
    #     mask_array[NonTargetRegion] = 0

    # 找到mask中非零部分的bounding box
    # non_zero_indices = np.argwhere(mask_array)
    range_indices = np.argwhere((mask_array >= start_index) & (mask_array <= end_index))

    min_z, min_y, min_x = range_indices.min(axis=0)
    max_z, max_y, max_x = range_indices.max(axis=0)

    # 扩展bounding box大小
    min_z = max(min_z - padding, 0)
    min_y = max(min_y - padding, 0)
    min_x = max(min_x - padding, 0)
    max_z = min(max_z + padding, mask_array.shape[0] - 1)
    max_y = min(max_y + padding, mask_array.shape[1] - 1)
    max_x = min(max_x + padding, mask_array.shape[2] - 1)

    # 提取扩展后的区域
    extracted_image_array = image_array[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]
    extracted_mask_array = mask_array[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

    # 先把原来1的地方置为0，然后再将大骨部分置为1
    if start_index != 1:
        extracted_mask_array[extracted_mask_array == 1] = 0
        extracted_mask_array[extracted_mask_array == 2] = 0

        extracted_mask_array[extracted_mask_array == start_index] = 1 
    else:
        extracted_mask_array[extracted_mask_array == start_index] = 1 
    # 然后将骨折部分置为2
    for i in range(start_index+1,end_index+1):
        # print(i)
        extracted_mask_array[extracted_mask_array == i] = 2 
    extracted_mask_array[(extracted_mask_array != 1) & (extracted_mask_array != 2)] = 0

    extracted_image = sitk.GetImageFromArray(extracted_image_array)
    extracted_mask = sitk.GetImageFromArray(extracted_mask_array)

    # 设置图像的方向、原点和间距与原图像一致
    extracted_image.SetDirection(image.GetDirection())
    extracted_image.SetOrigin(image.GetOrigin())
    extracted_image.SetSpacing(image.GetSpacing())

    extracted_mask.SetDirection(mask.GetDirection())
    extracted_mask.SetOrigin(mask.GetOrigin())
    extracted_mask.SetSpacing(mask.GetSpacing())

    # 保存提取后的图像和mask
    sitk.WriteImage(extracted_image, output_image_path)
    sitk.WriteImage(extracted_mask, output_mask_path)





if __name__ == '__main__':    
    # 示例调用
    ori_data_path = r'/home/nas/pzh/PENGWIN/task1/ori_mha_data'
    ori_label_path = r'/home/nas/pzh/PENGWIN/task1/ori_mha_label'

    data_root = r'/home/nas/pzh/PENGWIN/ttseg'

    # output_folder = '/media/siat/disk_2.0TB/pzh/data/PENGWIN/roi_data_new'
    output_folder = r'/home/nas/pzh/PENGWIN/task1/roi_data_new'

    output_hip_left_folder = os.path.join(output_folder, 'hip_left')
    output_hip_right_folder = os.path.join(output_folder, 'hip_right')
    output_sacurm_folder = os.path.join(output_folder, 'sacrum')
    os.makedirs(output_hip_left_folder, exist_ok=True)
    os.makedirs(output_hip_right_folder, exist_ok=True)
    os.makedirs(output_sacurm_folder, exist_ok=True)
    data_list = os.listdir(data_root)
    data_list = sorted(data_list)
    for index, data_folder in enumerate(data_list,start=1):
        id = str(index).zfill(3)
        # if id != '080':
        #     continue
        print(id)
        print(data_folder)

        image_path = os.path.join(ori_data_path, id+'.mha')
        # print(image_path)
        hip_left_mask_path = os.path.join(ori_label_path, id+'.mha')
        hip_right_mask_path = os.path.join(ori_label_path, id+'.mha')
        sacurm_mask_path = os.path.join(ori_label_path, id+'.mha')

        # hip_left_mask_path = os.path.join(data_root, data_folder,  'hip_left.nii.gz')
        # hip_right_mask_path = os.path.join(data_root, data_folder,  'hip_right.nii.gz')
        # sacurm_mask_path = os.path.join(data_root, data_folder,  'sacrum.nii.gz')


        output_hip_left_path = os.path.join(output_hip_left_folder, id+'_hip_left_roi.nii.gz')
        output_hip_right_path = os.path.join(output_hip_right_folder, id+'_hip_right_roi.nii.gz')
        output_sacurm_path = os.path.join(output_sacurm_folder, id+'_sacrum_roi.nii.gz')
        
        output_hip_left_mask_path = os.path.join(output_hip_left_folder, id+'_hip_left_mask.nii.gz')
        output_hip_right_mask_path = os.path.join(output_hip_right_folder, id+'_hip_right_mask.nii.gz')
        output_sacurm_mask_path = os.path.join(output_sacurm_folder, id + '_sacrum_mask.nii.gz')

        print(output_hip_left_path)
        # print(output_hip_left_mask_path)
        extract_region_with_padding(image_path, sacurm_mask_path, output_sacurm_path, output_sacurm_mask_path, padding=5, start_index = 1, end_index = 10)
        extract_region_with_padding(image_path, hip_left_mask_path, output_hip_left_path, output_hip_left_mask_path, padding=5, start_index = 11, end_index = 20)
        extract_region_with_padding(image_path, hip_right_mask_path, output_hip_right_path, output_hip_right_mask_path, padding=5, start_index = 21, end_index = 30)
        # break
