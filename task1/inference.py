import SimpleITK as sitk
import numpy as np
import os
from pathlib import Path
import glob
import SimpleITK
import numpy
import nibabel as nib
from resources.totalsegmentator.python_api import totalsegmentator
import torch
import scipy.ndimage as ndi
import time
from scipy.ndimage import label, distance_transform_edt

# from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join

# INPUT_PATH = Path("test/input")
# OUTPUT_PATH = Path("test/output")
# TEMP_PATH = Path("test/temp")
# RESOURCE_PATH = Path("resources")

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
TEMP_PATH = Path("temp")
RESOURCE_PATH = Path("resources")

TEMP_NIFTI_PATH = TEMP_PATH / "temp_nii"
INPUT_NIFTI_PATH = INPUT_PATH / "images/pelvic-fracture-ct"
OUTPUT_NIFTI_PATH = OUTPUT_PATH / "images/pelvic-fracture-ct-segmentation"

os.makedirs(TEMP_PATH, exist_ok=True)
os.makedirs(TEMP_NIFTI_PATH, exist_ok=True)


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def extract_region_with_padding(image_array, mask_array,padding=5,index=25):
     # 因为后续是单独推理，在取ROI的时候将另外两大骨区域的CT值置为背景可以防止网络误判骨折碎片
    image_temp = image_array.copy()

    # 找到mask中非零部分的bounding box
    # non_zero_indices = np.argwhere(mask_array)
    # 三大骨的区域的数组,在ttseg中骶骨索引是25，左髋骨是77，右髋骨是78

    range_indices = np.argwhere(mask_array == index)
    min_z, min_y, min_x = range_indices.min(axis=0)
    max_z, max_y, max_x = range_indices.max(axis=0)

    # 扩展bounding box大小
    min_z = max(min_z - padding, 0)
    min_y = max(min_y - padding, 0)
    min_x = max(min_x - padding, 0)
    max_z = min(max_z + padding, mask_array.shape[0] - 1)
    max_y = min(max_y + padding, mask_array.shape[1] - 1)
    max_x = min(max_x + padding, mask_array.shape[2] - 1)
    
    # nib的维度是xyz,这里变量的名字和实际的xyz的不一样
    # 提取扩展后的区域
    extracted_image_array = image_temp[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

    # extracted_mask_array = mask_array[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]
    # return extracted_image_array, extracted_mask_array

    # 用于最后恢复原图
    return extracted_image_array, [min_z, min_y, min_x, max_z, max_y, max_x]


def get_all_roi(nib_ct_img):
    # TTseg定位三大骨
    ct_arr = nib_ct_img.get_fdata()
    # 翻转成sitk的顺序
    ct_arr = np.transpose(ct_arr, (2, 1, 0))
     # 使用 TotalSegmentator 进行处理
    roi_list = ['hip_left', 'hip_right', 'sacrum']
    # option 2: provide input and output as nifti image objects
    output_img = totalsegmentator(nib_ct_img, roi_subset=roi_list)
    print('三大骨定位完成')

    # 三大骨的区域的数组,在ttseg中骶骨索引是25，左髋骨是77，右髋骨是78
    seg_arr = output_img.get_fdata()
    seg_arr = np.transpose(seg_arr, (2, 1, 0))
    
    hip_left_roi, hip_left_roi_info = extract_region_with_padding(ct_arr, seg_arr, padding=5, index=77)
    hip_right_roi, hip_right_roi_info = extract_region_with_padding(ct_arr, seg_arr, padding=5, index=78)
    sacrum_roi, sacrum_roi_info = extract_region_with_padding(ct_arr, seg_arr, padding=5, index=25)
    print('提取ROI完成')

    # print(hip_left_roi_info)
    # print(hip_right_roi_info)
    # print(sacrum_roi_info)

    return ct_arr.shape, [hip_left_roi, hip_right_roi, sacrum_roi],[hip_left_roi_info, hip_right_roi_info, sacrum_roi_info]


def nnUNet_predict(roi_data=None, properties=None, dataset='Dataset141_HipLeft',chk='checkpoint_best.pth'):
    # from resources.nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
    from resources.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


    nnUNet_source = RESOURCE_PATH
    # nnUNet预测
        # Set the environment variable to handle memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    torch.cuda.empty_cache()

    predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=False,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True)

    predictor.initialize_from_trained_model_folder(
        os.path.join(nnUNet_source, f'{dataset}/nnUNetTrainer_NexToU_ep500_CosAnneal__nnUNetPlans__3d_fullres_nextou'),
        use_folds='all',
        checkpoint_name=chk,
    )
    ret = predictor.predict_single_npy_array(input_image=np.expand_dims(roi_data, axis=0), 
                                                                            image_properties={'spacing':properties},
                                                                            segmentation_previous_stage=None, 
                                                                            output_file_truncated=None, 
                                                                            save_or_return_probabilities=None
                                                                            )
    
    return ret




def combine_masks_with_cca(ct_arr_shape, result_list, roi_info_list):
    result_arr = np.zeros(ct_arr_shape)
    hip_left_roi_info = roi_info_list[0]
    hip_right_roi_info = roi_info_list[1]
    sacrum_roi_info = roi_info_list[2]


    hip_left_arr = process_segmentation1(result_list[0], other_size_threshold=50)
    hip_right_arr = process_segmentation1(result_list[1], other_size_threshold=50)
    sacrum_arr = process_segmentation1(result_list[2], other_size_threshold=50)

    # 三大骨的区域的数组,在标签中骶骨索引是1-10，左髋骨是11-20，右髋骨是21-30
    # 找到 hip_left_arr 中非零的位置加 10
    hip_left_arr[hip_left_arr != 0] += 10
    # 找到 hip_right_arr 中非零的位置加 20
    hip_right_arr[hip_right_arr != 0] += 20

     # 打印数组形状以进行调试
    print("Hip Left ROI shape:", hip_left_arr.shape)
    print("Hip Right ROI shape:", hip_right_arr.shape)
    print("Sacrum ROI shape:", sacrum_arr.shape)

    #  合并标签
    result_arr[hip_left_roi_info[0]:hip_left_roi_info[3] + 1, 
               hip_left_roi_info[1]:hip_left_roi_info[4] + 1, 
               hip_left_roi_info[2]:hip_left_roi_info[5] + 1] = np.maximum(
        result_arr[hip_left_roi_info[0]:hip_left_roi_info[3] + 1, 
                   hip_left_roi_info[1]:hip_left_roi_info[4] + 1, 
                   hip_left_roi_info[2]:hip_left_roi_info[5] + 1],
        hip_left_arr
    )

    result_arr[hip_right_roi_info[0]:hip_right_roi_info[3] + 1, 
               hip_right_roi_info[1]:hip_right_roi_info[4] + 1, 
               hip_right_roi_info[2]:hip_right_roi_info[5] + 1] = np.maximum(
        result_arr[hip_right_roi_info[0]:hip_right_roi_info[3] + 1, 
                   hip_right_roi_info[1]:hip_right_roi_info[4] + 1, 
                   hip_right_roi_info[2]:hip_right_roi_info[5] + 1],
        hip_right_arr
    )

    result_arr[sacrum_roi_info[0]:sacrum_roi_info[3] + 1, 
               sacrum_roi_info[1]:sacrum_roi_info[4] + 1, 
               sacrum_roi_info[2]:sacrum_roi_info[5] + 1] = np.maximum(
        result_arr[sacrum_roi_info[0]:sacrum_roi_info[3] + 1, 
                   sacrum_roi_info[1]:sacrum_roi_info[4] + 1, 
                   sacrum_roi_info[2]:sacrum_roi_info[5] + 1],
        sacrum_arr
    )
    return result_arr


def write_result(nib_ct_img, uuid, result_list, roi_info_list):

    ct_arr = nib_ct_img.get_fdata()
    # 翻转成sitk的顺序
    ct_arr = np.transpose(ct_arr, (2, 1, 0))
    ct_arr_shape = ct_arr.shape
    
    # 023.mha调试用的信息
    # hip_left_roi_info = [np.int64(32), np.int64(20), np.int64(25), np.int64(311), np.int64(210), np.int64(192)]
    # hip_right_roi_info = [np.int64(34), np.int64(24), np.int64(187), np.int64(305), np.int64(225), np.int64(380)]
    # sacrum_roi_info = [np.int64(87), np.int64(12), np.int64(137), np.int64(284), np.int64(130), np.int64(279)]
    # result_arr = combine_masks_with_cca(ct_arr_shape, hip_left_roi_info, hip_right_roi_info, sacrum_roi_info)

    result_arr = combine_masks_with_cca(ct_arr_shape, result_list=result_list, roi_info_list=roi_info_list)

    # 确保结果数组的 dtype 为 int8
    result_arr = result_arr.astype(np.int8)
    print("Result array dtype:", result_arr.dtype)
    result_image = sitk.GetImageFromArray(result_arr)
    result_image.SetDirection(input_img.GetDirection())
    result_image.SetOrigin(input_img.GetOrigin())
    result_image.SetSpacing(input_img.GetSpacing())
    output_file_path = os.path.join(OUTPUT_NIFTI_PATH, uuid+'.mha')
    print(output_file_path)
    os.makedirs(os.path.join(OUTPUT_NIFTI_PATH), exist_ok=True)
    sitk.WriteImage(result_image, str(output_file_path),useCompression=True)
    print(result_arr.shape)

def process_segmentation1(seg_array, other_size_threshold=10):
    # 提取主要碎片和其他碎片
    main_fragment = (seg_array == 1)
    other_fragments = (seg_array == 2)

    # 创建一个新的掩码来存储处理后的结果
    processed_segmentation = np.zeros_like(seg_array)

    # 对主要碎片应用连通分量分析
    main_labeled_fragments, num_features = ndi.label(main_fragment)
    # 找到最大的连通域
    sizes = ndi.sum(main_fragment, main_labeled_fragments, range(1, num_features + 1))
    max_label = np.argmax(sizes) + 1  # +1 因为范围是从1到num_features
    # 保留主要碎片最大的连通域
    largest_main_fragment = (main_labeled_fragments == max_label)
    processed_segmentation[processed_segmentation != largest_main_fragment] = 0
    processed_segmentation[largest_main_fragment] = 1

    # 对其他碎片应用连通分量分析
    other_labeled_fragments, num_features = ndi.label(other_fragments)

    # 确定输入图像的维度
    dimensions = seg_array.ndim
    # 生成适合输入图像维度的结构元素
    structure_element = np.ones((3,) * dimensions)

    # 去除次要碎片中小的连通分量或进行值更改
    for i in range(1, num_features + 1):
        fragment = (other_labeled_fragments == i)
        # print(np.sum(fragment))
        if np.sum(fragment) < other_size_threshold:
            # 获取外围一圈的邻域体素
            dilated_fragment = ndi.binary_dilation(fragment, structure=structure_element)
            boundary = dilated_fragment & ~fragment
            
            # 获取邻域体素的值
            # 这里获取邻域体素的值的时候要选择已经找到最大连通域后的数组,不然就会和主要碎片的其他小连通域一起计算了
            neighbor_values = processed_segmentation[boundary]
            unique_values, counts = np.unique(neighbor_values, return_counts=True)
            # 找到出现次数最多的值，包括背景值
            most_common_value = unique_values[np.argmax(counts)]
            # 如果邻域体素全是背景
            if len(unique_values) == 1 and unique_values[0] == 0:
                # 即将小的次要碎片变成邻域的值
                # 去除游离的小碎片
                continue  
            else:
                # 将连通分量值变为邻域中出现次数最多的值
                # 即将小的次要碎片变成邻域的值
                processed_segmentation[fragment] = most_common_value


        # 即保留该大的次要碎片
        else:
            processed_segmentation[fragment] = 2

    # 重新标记其他碎片的连通分量
    relabelled_fragments, new_num_features = ndi.label(processed_segmentation == 2)

    # 将重新标记的碎片添加回处理后的结果
    for i in range(1, new_num_features + 1):
        processed_segmentation[relabelled_fragments == i] = i + 1

    # 计算指定范围内每个连通域的最近距离,大于一定距离的置0
    processed_segmentation = update_segmentation(processed_segmentation)

    # 最后按大小排序，因为最多只有10个碎片，所以按照连通域的大小重新排序，然后去掉第11个碎片之后的（连通域）
    sorted_segmentation = create_sorted_segmentation(processed_segmentation)


    return sorted_segmentation


def create_sorted_segmentation(segmentation):
    # 创建一个全0的数组，大小与processed_segmentation相同
    sorted_segmentation = np.zeros_like(segmentation)
    
    # 获取最大值
    max_value = segmentation.max()
    
    # 用于存储连通域大小和其对应的标签
    regions = []

    # 遍历值为2到最大值的所有连通域
    for value in range(2, max_value + 1):
        current_mask = (segmentation == value)
        
        if np.any(current_mask):
            labeled_array, num_features = label(current_mask)
            
            for region_label in range(1, num_features + 1):
                region_mask = (labeled_array == region_label)
                region_size = region_mask.sum()
                regions.append((region_size, region_mask))

    # 按连通域大小排序，降序排列
    regions.sort(reverse=True, key=lambda x: x[0])
    
    # 重新赋值，连通域越大，值越小，从2开始
    new_value = 2
    for _, region_mask in regions:
        if new_value <= 10:
            sorted_segmentation[region_mask] = new_value
        new_value += 1
    
    # 将 processed_segmentation 中值为1的地方赋值给 sorted_segmentation
    sorted_segmentation[segmentation == 1] = 1
    
    return sorted_segmentation


def update_segmentation(segmentation, start=2, end=10, distance_threshold=15):
    updated_segmentation = segmentation.copy()
    
    for value in range(start, end + 1):
        # 创建当前值的掩码
        current_mask = (segmentation == value)
        
        if np.any(current_mask):
            # 识别当前值的连通域
            labeled_array, num_features = label(current_mask)

            for region_label in range(1, num_features + 1):
                region_mask = (labeled_array == region_label)

                # 创建其他非0连通域的掩码
                other_mask = (segmentation > 0) & ~region_mask

                # 计算距离变换
                distance = distance_transform_edt(~other_mask)

                # 获取当前连通域的最小距离
                min_distance = distance[region_mask].min()

                # 如果距离大于阈值，将该连通域置为0
                if min_distance > distance_threshold:
                    updated_segmentation[region_mask] = 0
                    print(f"值为 {value} 的连通域 {region_label} 被置为0，距离: {min_distance}")

    return updated_segmentation

def process_segmentation2(seg_array, other_size_threshold=10):
    # 提取主要碎片和其他碎片
    main_fragment = (seg_array == 1)
    other_fragments = (seg_array == 2)

    # 创建一个新的掩码来存储处理后的结果
    processed_segmentation = np.zeros_like(seg_array)

    # 对主要碎片应用连通分量分析
    main_labeled_fragments, num_features = ndi.label(main_fragment)
    # 找到最大的连通域
    sizes = ndi.sum(main_fragment, main_labeled_fragments, range(1, num_features + 1))
    max_label = np.argmax(sizes) + 1  # +1 因为范围是从1到num_features
    # 保留主要碎片最大的连通域
    largest_main_fragment = (main_labeled_fragments == max_label)
    processed_segmentation[processed_segmentation != largest_main_fragment] = 0
    processed_segmentation[largest_main_fragment] = 1

    # 对其他碎片应用连通分量分析
    other_labeled_fragments, num_features = ndi.label(other_fragments)

    # 去除次要碎片中小的连通分量或进行值更改
    for i in range(1, num_features + 1):
        fragment = (other_labeled_fragments == i)
        # print(np.sum(fragment))
        if np.sum(fragment) < other_size_threshold:
            continue  
        # 即保留该大的次要碎片
        else:
            processed_segmentation[fragment] = 2

    # 重新标记其他碎片的连通分量
    relabelled_fragments, new_num_features = ndi.label(processed_segmentation == 2)

    # 将重新标记的碎片添加回处理后的结果
    for i in range(1, new_num_features + 1):
        processed_segmentation[relabelled_fragments == i] = i + 1

    return processed_segmentation


if __name__ == "__main__":
    # raise SystemExit(run())
    start_time = time.time()

    # Your code here

    _show_torch_cuda_info()
    print(INPUT_NIFTI_PATH)
    print("Directory contents:", os.listdir(INPUT_NIFTI_PATH))
    input_file = glob.glob(str(INPUT_NIFTI_PATH / "*.mha"))[0]
    uuid = os.path.basename(os.path.splitext(input_file)[0])
    print(input_file)
    print(uuid)
    
    # 使用 SimpleITK 读取.mha 文件,并保存成.nii.gz文件
    input_img = sitk.ReadImage(input_file)
    properties = input_img.GetSpacing()
    sitk.WriteImage(input_img, str(os.path.join(TEMP_NIFTI_PATH, uuid+'.nii.gz')))
    nib_ct_img = nib.load(str(os.path.join(TEMP_NIFTI_PATH, uuid+'.nii.gz')))

    # 获取三大骨的ROI区域
    # ct_arr_shape, hip_left_roi_info, hip_right_roi_info, sacrum_roi_info = get_all_roi(nib_ct_img)
    ct_arr_shape, roi_data_list, roi_info_list = get_all_roi(nib_ct_img)


    # 分别推理骨折碎片
    HipLeft_result = nnUNet_predict(roi_data_list[0], properties, dataset='Dataset141_HipLeft',chk='checkpoint_best.pth')
    HipLeft_result = HipLeft_result.squeeze()
    print("HipLeft推理完成")
    HipRight_result = nnUNet_predict(roi_data_list[1], properties, dataset='Dataset142_HipRight',chk='checkpoint_best.pth')
    HipRight_result = HipRight_result.squeeze()
    print("HipRight推理完成")
    Sacrum_result = nnUNet_predict(roi_data_list[2], properties, dataset='Dataset143_Sacrum',chk='checkpoint_best.pth')
    Sacrum_result = Sacrum_result.squeeze()
    print("Sacrum推理完成")

    # 后处理并保存
    write_result(nib_ct_img, uuid, [HipLeft_result, HipRight_result, Sacrum_result], roi_info_list)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")





