import SimpleITK as sitk
import numpy as np
import skimage as ski
from skimage import measure


def getDSC(SR, GT):
    """
    3维计算DSC，输入都是二值图，格式是array
    """
    Inter = np.sum(((SR + GT) == 2).astype(np.float32))
    DC = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)
    return DC


def get_detection_result(SR, GT, detect_threshold=0.5):
    """
    3维计算检测指标，precision/recall/f1-score
    其中precision同PPV，recall同sensitivity，在检测中无法计算specificity
    :param:detect_threshold:重合程度多少认为是正确检测到
    """
    tp1, fp1, fn1 = 0, 0, 0  # tn在检测中是无意义的
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)

    _input = sitk.GetImageFromArray(SR.astype(np.uint8))
    output_ex = cca.Execute(_input)
    # pre_labeled = sitk.GetArrayFromImage(output_ex)
    num_pre = cca.GetObjectCount()
    # for ii in range(1, num_pre + 1):
    #     pre_one = pre_labeled == ii  # 取到其中一个连通域
    #     cover_area = pre_one * GT  # 和金标准的重合区域
    #     if np.sum(cover_area) / np.sum(pre_one) >= detect_threshold:  # 重合率大于阈值
    #         tp1 += 1
    #     else:
    #         fp1 += 1
    _input = sitk.GetImageFromArray(GT.astype(np.uint8))
    output_ex = cca.Execute(_input)
    gt_labeled = sitk.GetArrayFromImage(output_ex)
    num_gt = cca.GetObjectCount()
    for ii in range(1, num_gt + 1):
        gt_one = gt_labeled == ii  # 取到其中一个连通域
        cover_area = gt_one * SR  # 该连通域和预测结果的重合区域
        if np.sum(cover_area) / np.sum(gt_one) >= detect_threshold:  # 重合率大于阈值
            tp1 += 1
        else:
            fn1 += 1
    fp1 = num_pre - tp1
    if fp1 < 0:     # 防止一个病灶里有多个P的情况
        fp1 = 0
    # 得到每例结果
    precision1 = tp1 / (tp1 + fp1 + 1e-6)
    recall1 = tp1 / (tp1 + fn1 + 1e-6)
    f1_score1 = 2 * precision1 * recall1 / (precision1 + recall1 + 1e-6)
    return [precision1, recall1, f1_score1, tp1, fp1, fn1]


def check_componnent_3d(mask, pet=None, min_size=100, max_size=1e8):
    """
    检查全身分割结果，去掉不符合的病灶
    """
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    # cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)
        if np.sum(label_temp[:]) <= min_size:  # 体积太小
            mask[label_temp] = 0
            # print(np.sum(label_temp[:]))
            continue
        if np.sum(label_temp[:]) >= max_size:  # 体积太大
            mask[label_temp] = 0
            # print(np.sum(label_temp[:]))
            continue
        # 获取区域在z轴上的范围，如果不连续则去掉
        label_temp_z = np.sum(np.sum(label_temp, axis=2), axis=1)
        label_temp_z = label_temp_z > 0
        if np.sum(label_temp_z) <= 1:  # z轴上不连续
            mask[label_temp] = 0
            continue
        if pet is not None:
            if np.max(pet[label_temp][:]) <= 0.2:  # 均值太小
                mask[label_temp] = 0
                continue
    return mask.astype(np.uint8)


def check_componnent_2d(mask, min_size=10):
    # 检查连通域(2d)
    for j in range(mask.shape[0]):
        label_t, num_t = ski.measure.label(mask[j, :, :], connectivity=2, return_num=True)
        for jj in range(1, num_t + 1):
            label_temp_t = (label_t == jj)
            if np.sum(label_temp_t[:]) <= min_size:  # 体积太小
                mask[j, label_temp_t] = 0
    return mask


def check_componnent_organ(mask, organ_mask):
    """
    检查全身分割结果，去掉在膀胱、肾脏、脑部内的
    """
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)
        if np.sum(label_temp[organ_mask == 2]) > 0 or np.sum(label_temp[organ_mask == 3]) > 0 or \
                np.sum(label_temp[organ_mask == 8]) > 0 or np.sum(label_temp[organ_mask == 9]) > 0:  #
            mask[label_temp] = 0
    return mask


def check_liver(mask, organ_mask, PET_array):
    """
       检查全身分割结果，去掉SUVmax小于肝脏SUVmax
    """
    if np.max(mask) == 0:
        return mask
    liver_mask = organ_mask == 6
    # 获取肺部区域的质心
    labeled_img, num = measure.label(liver_mask, connectivity=3, return_num=True)
    liver_centroid = np.array(measure.regionprops(labeled_img)[0].centroid, dtype=np.int)  # z,y,x
    # 在一个范围内取最大值，作为肝本底的SUVmax
    liver_suv = np.max(PET_array[liver_centroid[0]-5:liver_centroid[0]+5, liver_centroid[1]-5:liver_centroid[1]+5,
                       liver_centroid[2]-5:liver_centroid[2]+5])
    # 防止SUV太高的情况出现
    if liver_suv > 9:
        print(liver_suv)
        return mask
    # 如果检测到的所有病灶的max小于肝SUVmax，则不进行与肝本底的比较
    if np.max(PET_array[np.array(mask, dtype=np.bool)]) < liver_suv:
        print('low SUV in lesion')
        return mask
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)
        if np.max(PET_array[label_temp]) < liver_suv:
            mask[label_temp] = 0
    return mask