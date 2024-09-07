import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像（大小也与目标图像一致）
    使用示范：
    from img_resize_zsn import resize_image_itk
    target_img = sitk.ReadImage(target_img_file)
    ori_img = sitk.ReadImage(ori_img_file)
    img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
    注意事项：
    ori_img不能是直接从array转成sitk的图像，这会导致结果是一片空白
    """
    target_Size = target_img.GetSize()  # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()  # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值是用于mask的，所以保存uint8格式
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值是用于PET/CT/MRI之类的，所以保存float32格式
    resampler.SetTransform(sitk.Transform())        # 3, sitk.sitkIdentity这个参数的用处还不确定
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled


def resample_image(itkimage, newspacing=(0.78515625, 0.78515625, 1.25), resamplemethod=sitk.sitkLinear):
    print('--resize ing--')
    resampler = sitk.ResampleImageFilter()
    oridirection = itkimage.GetDirection()
    orisize = itkimage.GetSize()
    orispacing = itkimage.GetSpacing()
    factor = np.array(newspacing)/np.array(orispacing)
    newsize = (np.around(np.array(orisize) / factor)).astype(int)
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newsize.tolist())
    resampler.SetOutputSpacing(newspacing)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_res = resampler.Execute(itkimage)  # 得到重新采样后的图像
    array_img_res = sitk.GetArrayFromImage(itk_img_res)
    array_img_crop = array_img_res[:-1,
                     int(newsize[1] / 2) - int(orisize[1] / 2):int(newsize[1] / 2) + int(orisize[1] / 2),
                     int(newsize[0] / 2) - int(orisize[0] / 2):int(newsize[0] / 2) + int(orisize[0] / 2)]
    array_img_crop[0, 0, 0] = 3000
    itk_img_crop = sitk.GetImageFromArray(array_img_crop)
    itk_img_crop.SetDirection(oridirection)
    itk_img_crop.SetSpacing(newspacing)
    print('--resize finish--')
    return itk_img_crop


def resample(imgs, new_shape, order=3):
    """
    输入输出是array
    功能：输入新的shape进行resample
    order-样条插值的次数，范围0-5，一般用到0和3: 0-最近邻插值、1-线性插值、2-线性插值、3-三次插值
    """
    if len(imgs.shape) == 3:
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs


# # 初始版本,from网上,适用于只有spacing不同的两个图
# def resize_image(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
#     print('--resize ing--')
#     resampler = sitk.ResampleImageFilter()
#     originSize = itkimage.GetSize()  # 原来的体素块尺寸
#     originSpacing = itkimage.GetSpacing()
#
#     newSize = np.array(newSize, float)
#     factor = originSize / newSize
#     newSpacing = originSpacing * factor
#     newSize = newSize.astype(np.int)    # spacing肯定不能是整数
#
#     resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
#     resampler.SetSize(newSize.tolist())
#     resampler.SetOutputSpacing(newSpacing.tolist())
#     resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
#     resampler.SetInterpolator(resamplemethod)
#     itk_img_res = resampler.Execute(itkimage)  # 得到重新采样后的图像
#     print('--resize finish--')
#     return itk_img_res
