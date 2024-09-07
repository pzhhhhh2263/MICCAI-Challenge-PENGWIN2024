import h5py
import cv2
import pengwin_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from UNet import AttU_Net
from xray_config_test import config
import torch
import os
from torchvision import transforms as T
import scipy
import tifffile
from pathlib import Path
import glob

# INPUT_PATH = Path("test/input")
# OUTPUT_PATH = Path("test/output")
# TEMP_PATH = Path("test/temp")
# RESOURCE_PATH = Path("resources")

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.paths import nnUNet_results, nnUNet_raw
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
TEMP_PATH = Path("temp")
RESOURCE_PATH = Path("resources")

# 改成xray！！！！！！！！！！！！！！！！！！！！！！！！！
TEMP_NIFTI_PATH = TEMP_PATH / "temp_nii"
INPUT_NIFTI_PATH = INPUT_PATH / "images/pelvic-fracture-ct"
OUTPUT_NIFTI_PATH = OUTPUT_PATH / "images/pelvic-fracture-ct-segmentation"



os.makedirs(TEMP_PATH, exist_ok=True)
os.makedirs(TEMP_NIFTI_PATH, exist_ok=True)





'''definition'''
def extract_region_with_padding(tif_image,mask_image, padding=5):
    # # 将图像和mask转换为numpy数组
    # image_array = image
    # mask_array = mask

    # 找到mask中非零部分的bounding box
    non_zero_indices = np.argwhere(mask_image)
    # range_indices = np.argwhere((mask_array >= start_index) & (mask_array <= end_index))

    min_y, min_x = non_zero_indices.min(axis=0)
    max_y, max_x = non_zero_indices.max(axis=0)
    # print(min_y, min_x, max_y, max_x)
    # 扩展bounding box大小
    min_y = max(min_y - padding, 0)
    min_x = max(min_x - padding, 0)
    max_y = min(max_y + padding, mask_image.shape[0] - 1)
    max_x = min(max_x + padding, mask_image.shape[1] - 1)

    # 提取扩展后的区域
    extracted_image_array = tif_image[min_y:max_y + 1, min_x:max_x + 1]
    roi_info = [min_y, min_x, max_y, max_x]

    return extracted_image_array,roi_info

def resize_with_interpolation(image, target_shape):  # 利用双线性插值缩放图像
    resized_image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return resized_image


def read_tif(image_path):  # 读取tiff文件

    # tif数据
    tif_image = Image.open(image_path)
    tif_image = np.array(tif_image)


    return  tif_image

def get_IDs(resized_to_original):
    # load image and masks
    # image,mask = read_tif(image_path,seg_path) # raw intensity image
    masks, category_ids, fragment_ids = pengwin_utils.load_masks(resized_to_original)
    pred_masks, pred_category_ids, pred_fragment_ids = masks, category_ids, fragment_ids
    return pred_masks, pred_category_ids, pred_fragment_ids


def remove_non_channel(pred_img):
    # 找出非空通道，保留非空通道
    non_empty_channels = [i for i in range(pred_img.shape[0]) if np.any(pred_img[i]==1)]
    filtered_result = pred_img[non_empty_channels]

    # 打印非空通道数
    print("非空通道数:", len(non_empty_channels))

    # 打印非空通道索引
    print("非空通道索引:", non_empty_channels)

    return filtered_result

def resize_to_original(image, original_size):
    h, w = image.shape[:2]
    original_h, original_w = original_size[:2]

    # 计算缩放比例
    scale_h = original_h / h
    scale_w = original_w / w

    # 计算缩放后的尺寸
    new_h = int(h * scale_h)
    new_w = int(w * scale_w)

    # 缩放图像
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_image


def hu_clipping_and_normalization(data, percentile_00_5=0.03209742857143283, percentile_99_5=4.22320652246475):
    # Step 1: HU Clipping
    clipped_data = np.clip(data, percentile_00_5, percentile_99_5)

    # Step 2: Normalization to 0-1
    normalized_data = (clipped_data - percentile_00_5) / (percentile_99_5 - percentile_00_5)

    return normalized_data

def test_net(extracted_image_448):
    # ============== 参数设置 ===================
    # 设置随机数种子
    seed = config.seed
    np.random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 是否使用设定好的分折信息
    fold_K = config.fold_K
    fold_idx = config.fold_idx
    Task_name = config.Task_name

    # 路径设置
    result_path = config.result_path
    save_path = os.path.join(result_path, Task_name + '_fold' + str(fold_K) + '-' + str(fold_idx))  # 结果保存地址
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # 模型保存地址
    model_path = config.result_path
    weight_c = os.path.join(model_path, 'best.pkl')

    # 设置GPU相关
    cuda_idx = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)
    device = torch.device('cuda')

    with torch.no_grad():
        print('loading model from ' + weight_c)
        # 构建模型
        # 用smp构建网络
        model = eval(config.model_name)(encoder_name=config.encoder_name, encoder_weights=None,
                                        in_channels=config.img_ch, classes=config.output_ch)
        # print(model)
        model.to(device)
        best_status = torch.load(weight_c)
        model.load_state_dict(best_status['model'])  # , False
        model.eval()
        #归一化
        xray_array_nor = hu_clipping_and_normalization(extracted_image_448)
        #转张量
        Transform_ = T.Compose([T.ToTensor()])
        xray_tensor = Transform_(xray_array_nor)
        xray_tensor = torch.unsqueeze(xray_tensor, 0)
        xray_tensor = xray_tensor.to(device)
        #测试
        seg_result = model(xray_tensor.float())
        seg_result = torch.sigmoid(seg_result)
        seg_result = seg_result > 0.5
        seg_result = (torch.squeeze(seg_result)).data.cpu().numpy().astype(np.uint8)
        print(seg_result.shape)

    #去掉空的通道
    filtered_result = remove_non_channel(seg_result)
    # 将(448，488)缩小到原大小
    resized_to_original = np.array([resize_to_original(ch, (roi_bbox[3], roi_bbox[2])) for ch in filtered_result])
    # 补0到 (10, 448, 448)
    padded_pred_img = pad_to_shape(resized_to_original, (10, 448, 448), roi_bbox)
    return padded_pred_img

def pad_to_shape(image, target_shape, roi_bbox):
    padded_image = np.zeros(target_shape, dtype=image.dtype)
    x, y, w, h = roi_bbox
    padded_image[:, y:y + h, x:x + w] = image
    return padded_image

def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == '__main__':
    # image_path = "./xray/042_0133_normalized.tif"

    print("Directory contents:", os.listdir(INPUT_NIFTI_PATH))
    input_file = glob.glob(str(INPUT_NIFTI_PATH / "*.tiff"))[0]
    uuid = os.path.basename(os.path.splitext(input_file)[0])
    print(input_file)
    print(uuid)

    # ------------------------nnunet的一阶段定位------------------------------------
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=False,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True)
    
    checkpoint_name='checkpoint_final.pth'    
    predictor.initialize_from_trained_model_folder(
        os.path.join(nnUNet_source, 'Dataset301_CTA/nnUNetTrainer__nnUNetPlans__3d_fullres'),
        use_folds=(0, ),
        checkpoint_name=checkpoint_name,
    )
    predictor.dataset_json['file_ending'] = '.mha'
    images, properties = SimpleITKIO().read_images([ct_mha])
    predictor.predict_single_npy_array(images, properties, None, output_file_path, False)
    print('Prediction finished')






    #提取扩展后的区域(形状不统一)
    extracted_image_array, resized_roi, roi_bbox = extract_region_with_padding(tif_image, mask_image,padding=5)
    #将扩展后的区域zoom到(10，448，448)
    target_size = (448, 448)
    extracted_image_448 = resize_with_interpolation(extracted_image_array,target_size)
    roi_448 =resize_with_interpolation(resized_roi,target_size)  #用于计算定量指标
    '''预测'''
    device = torch.device('cuda', 0)
    net = AttU_Net()
    #三个类型分别预测
    pred_img_sacrum = test_net(extracted_image_448)
    pred_img_hip_left = test_net(extracted_image_448)
    pred_img_hip_right = test_net(extracted_image_448)
    #获取每一类的类别ID和碎片ID
    pred_masks1, pred_category_ids1, pred_fragment_ids1 = get_IDs(pred_img_sacrum)
    pred_masks2, pred_category_ids2, pred_fragment_ids2 = get_IDs(pred_img_hip_left)
    pred_masks3, pred_category_ids3, pred_fragment_ids3 = get_IDs(pred_img_hip_right)
    # 将三个结果结合
    pred_masks = np.concatenate((pred_masks1, pred_masks2, pred_masks3), axis=0)
    pred_category_ids = np.concatenate((pred_category_ids1, pred_category_ids2, pred_category_ids3), axis=0)
    pred_fragment_ids = np.concatenate((pred_fragment_ids1, pred_fragment_ids2, pred_fragment_ids3), axis=0)
    #得到最后的结果
    pred_seg = pengwin_utils.masks_to_seg(pred_masks, pred_category_ids, pred_fragment_ids)
    print(pred_seg.shape)
    #保存分割结果为tiff
    pred_seg_dir = r''
    if not os.path.exists(pred_seg_dir):
        os.makedirs(pred_seg_dir)
    # pred_seg_save_path = os.path.join(pred_seg_dir, '001_0000.tif')  # ensure dir exists!
    # print(pred_seg_save_path)
    # Image.fromarray(pred_seg).save(pred_seg_save_path)
    # print(f"Wrote segmentation to {pred_seg_save_path}")
    # print(os.listdir(''))

    # 将图像数据保存为 TIFF 文件，并包含元数据
    output_path = os.path.join(pred_seg_dir, '001_0000_with_metadata.tif')
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(pred_seg, photometric='minisblack', metadata={'spacing': 1, 'unit': 'um'},
                  resolution=(1, 1, 'CENTIMETER'))
    print(f"Wrote segmentation with metadata to {output_path}")
    print(os.listdir(pred_seg_dir))














