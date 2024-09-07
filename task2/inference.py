import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pengwin_utils
import numpy as np
from PIL import Image
from xray_config_test import config
import torch

from torchvision import transforms as T
import tifffile
from pathlib import Path
import glob
import SimpleITK as sitk
import scipy.ndimage
import segmentation_models_pytorch as smp
import cv2
import time

# INPUT_PATH = Path("test/input")
# OUTPUT_PATH = Path("test/output")
# RESOURCE_PATH = Path("resources")


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

INPUT_NIFTI_PATH = INPUT_PATH / "images/pelvic-fracture-x-ray"
OUTPUT_NIFTI_PATH = OUTPUT_PATH / "images/pelvic-fracture-x-ray-segmentation"


os.makedirs(OUTPUT_NIFTI_PATH, exist_ok=True)


# 设置GPU相关
cuda_idx = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)
device = torch.device('cuda')



def remove_non_channel(pred_img):
    # 找出非空通道，保留非空通道
    non_empty_channels = [i for i in range(pred_img.shape[0]) if np.any(pred_img[i]==1)]
    filtered_result = pred_img[non_empty_channels]
    # 打印非空通道数
    print("非空通道数:", len(non_empty_channels))
    # 打印非空通道索引
    print("非空通道索引:", non_empty_channels)

    return filtered_result


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

def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def only1stage(xray_arr, model_path,dataset='HipLeft'):
    # 三个类型分别预测
     # ============== 参数设置 ===================
    # 设置随机数种子
    seed = config.seed
    np.random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 模型保存地址
    if dataset=='HipLeft':
        weight_c = os.path.join(model_path, 'only1stage_HipLeft')
        weight_c = os.path.join(weight_c, '0811_model/epoch30.pkl')
        # weight_c = os.path.join(weight_c, '0811_model/epoch20.pkl')

    elif dataset=='HipRight':
        weight_c = os.path.join(model_path, 'only1stage_HipRight')
        weight_c = os.path.join(weight_c, '0811_model/epoch30.pkl')
        # weight_c = os.path.join(weight_c, '0811_model/epoch20.pkl')

    elif dataset=='Sacrum':
        weight_c = os.path.join(model_path, 'only1stage_Sacrum')
        weight_c = os.path.join(weight_c, '0811_model/epoch30.pkl')
        # weight_c = os.path.join(weight_c, '0811_model/epoch20.pkl')


    with torch.no_grad():
        print('loading model from ' + weight_c)
        # 构建模型
        # 用smp构建网络

        model = eval(config.model_name)(encoder_name=config.encoder_name, encoder_weights=None,
                                        in_channels=config.img_ch, classes=config.output_ch)
        print("Bulid model with " + config.model_name)
        # print(model)
        model.to(device)
        best_status = torch.load(weight_c)
        model.load_state_dict(best_status['model'])  # , False
        model.eval()

        #z-score标准化
        if dataset == 'HipLeft':
            xray_array_nor = zscore_normalize_image(xray_arr)
        elif dataset == 'HipRight':
            xray_array_nor = zscore_normalize_image(xray_arr)
        elif dataset == 'Sacrum':
            xray_array_nor = zscore_normalize_image(xray_arr)

        # 转张量
        Transform_ = T.Compose([T.ToTensor()])
        xray_tensor = Transform_(xray_array_nor)
        xray_tensor = torch.unsqueeze(xray_tensor, 0)
        xray_tensor = xray_tensor.to(device)
        # 测试
        seg_result = model(xray_tensor.float())
        seg_result = torch.sigmoid(seg_result)
        # 卡阈值
        seg_result = seg_result > 0.5
        seg_result = (torch.squeeze(seg_result)).data.cpu().numpy().astype(np.uint8)
        print(seg_result.shape)

    #去掉空的通道
    seg_result = remove_non_channel(seg_result)
    

    return seg_result


def retain_largest_connected_component(mask):
    """
    保留2D掩码中最大的连通域。

    参数:
    mask (numpy.ndarray): 2D二值掩码，其中0表示背景，1表示目标。

    返回:
    numpy.ndarray: 只保留最大连通域后的掩码。
    """
    # 查找所有连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # stats[:, 4] 是连通域的面积（即像素数量）
    # 忽略背景的连通域（即索引0），找到最大的连通域
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # 创建一个新的掩码，只保留最大的连通域
    largest_component_mask = (labels == max_label).astype(np.uint8)
    
    return largest_component_mask

def process_channels(pred_img):
    # 对每个通道进行处理
    for i in range(pred_img.shape[0]):
        pred_img[i] = retain_largest_connected_component(pred_img[i])
    return pred_img

if __name__ == '__main__':
    start_time = time.time()
    # _show_torch_cuda_info()
    print("Directory contents:", os.listdir(INPUT_NIFTI_PATH))
    xray_file = glob.glob(str(INPUT_NIFTI_PATH / "*.tif"))[0]
    uuid = os.path.basename(os.path.splitext(xray_file)[0])
    xray_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(xray_file)))

    # -----------------------------------------只有一个阶段----------------------------------------

    # -------------------------------------------骶骨-------------------------------------------
    # 预测结果
    sacrum_pred_result = only1stage(xray_arr, model_path=RESOURCE_PATH,dataset='Sacrum')
    # 对每个通道取最大连通域处理
    sacrum_pred_result = process_channels(sacrum_pred_result)
    # 类别ID
    sacrum_pred_category_ids = [1] * sacrum_pred_result.shape[0]
    # 碎片ID
    sacrum_pred_fragment_ids = list(range(1, sacrum_pred_result.shape[0] + 1))

    # -------------------------------------------左髋骨-------------------------------------------

    # 预测结果
    hipleft_pred_result =only1stage(xray_arr, model_path=RESOURCE_PATH,dataset='HipLeft')
    # 对每个通道取最大连通域处理
    hipleft_pred_result = process_channels(hipleft_pred_result)
    # 类别ID
    hipleft_pred_category_ids = [2] * hipleft_pred_result.shape[0]
    # 碎片ID
    hipleft_pred_fragment_ids = list(range(1, hipleft_pred_result.shape[0] + 1))


    # -------------------------------------------右髋骨-------------------------------------------
    # 预测结果
    hipright_pred_result =only1stage(xray_arr, model_path=RESOURCE_PATH,dataset='HipRight')
    # 对每个通道取最大连通域处理
    hipright_pred_result = process_channels(hipright_pred_result)
    # 类别ID
    hipright_pred_category_ids = [3] * hipright_pred_result.shape[0]
    # 碎片ID
    hipright_pred_fragment_ids = list(range(1, hipright_pred_result.shape[0] + 1))

    # ========================================合并三个的结果=======================================

    # 初始化合并结果的列表
    pred_masks_list = []
    pred_category_ids = []
    pred_fragment_ids = []

    # 合并非空的预测结果
    if sacrum_pred_result.size > 0:
        pred_masks_list.append(sacrum_pred_result)
        pred_category_ids.extend(sacrum_pred_category_ids)
        pred_fragment_ids.extend(sacrum_pred_fragment_ids)
        
    if hipleft_pred_result.size > 0:
        pred_masks_list.append(hipleft_pred_result)
        pred_category_ids.extend(hipleft_pred_category_ids)
        pred_fragment_ids.extend(hipleft_pred_fragment_ids)

    if hipright_pred_result.size > 0:
        pred_masks_list.append(hipright_pred_result)
        pred_category_ids.extend(hipright_pred_category_ids)
        pred_fragment_ids.extend(hipright_pred_fragment_ids)

    desired_shape = (1, 448, 448)

    # 合并所有非空的预测结果
    if len(pred_masks_list) > 0:
        pred_masks = np.concatenate(pred_masks_list, axis=0)
    else:
        pred_masks = np.empty((0, *desired_shape))  # 如果所有预测结果都为空，则返回空数组

    
    # 打印掩码形状和类别、碎片 ID
    print("掩码形状:", pred_masks.shape)
    print("类别 ID:", pred_category_ids)
    print("碎片 ID:", pred_fragment_ids)

    # 将掩码转换为 uint32 类型
    pred_masks = pred_masks.astype(np.uint32)

    # 得到最后的结果
    pred_seg = pengwin_utils.masks_to_seg(pred_masks, pred_category_ids, pred_fragment_ids)

    # 保存可视化
    vis_image = pengwin_utils.visualize_sample(xray_arr, pred_masks, pred_category_ids, pred_fragment_ids)
    vis_path = os.path.join(OUTPUT_NIFTI_PATH, uuid+ '.png')
    Image.fromarray(vis_image).save(vis_path)
    print(f"Wrote visualization to {vis_path}")
    

    # 将图像数据保存为 TIFF 文件，并包含元数据
    output_path = os.path.join(OUTPUT_NIFTI_PATH, uuid+ '.tif')
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(pred_seg, photometric='minisblack', metadata={'spacing': 1, 'unit': 'um'},
                  resolution=(1, 1, 'CENTIMETER'))
    print(f"Wrote segmentation with metadata to {output_path}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")












