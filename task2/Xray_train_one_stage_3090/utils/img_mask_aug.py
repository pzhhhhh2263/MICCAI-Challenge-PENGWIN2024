import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa


def data_aug(imgs, masks):  # 输入图像和标签,输入输出格式为numpy
    # 标准化格式
    imgs = np.array(imgs)
    masks = np.array(masks).astype(np.uint8)

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 设定随机函数

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # 50%图像进行水平翻转
            iaa.Flipud(0.5),  # 50%图像做垂直翻转

            sometimes(iaa.Crop(percent=(0, 0.1))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%

            sometimes(iaa.Affine(  # 对一部分图像做仿射变换
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
                rotate=(-45, 45),  # 旋转±45度之间
                shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
                order=[0, 1],  # 使用最邻近差值或者双线性差值
                cval=(0, 255),
                mode=ia.ALL,  # 边缘填充,随机0到255间的值
            )),

            # 使用下面的0个到3个之间的方法去增强图像
            iaa.SomeOf((0, 2),
                       [
                           # 锐化处理
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.85, 1.15)),

                           # 扭曲图像的局部区域
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),

                           # 改变对比度
                           iaa.contrast.LinearContrast((0.85, 1.15), per_channel=0.5),

                           # 用高斯模糊，均值模糊，中值模糊中的一种增强(图像不是255的不用,不然会出错)
                           # iaa.OneOf([
                           #     iaa.GaussianBlur((0, 3.0)),
                           #     iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                           #     iaa.MedianBlur(k=(3, 11)),
                           # ]),

                           # 加入高斯噪声(图像不是255的不用)
                           # iaa.AdditiveGaussianNoise(
                           #     loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           # ),

                           # 边缘检测，将检测到的赋值0或者255然后叠在原图上(不确定)
                           # sometimes(iaa.OneOf([
                           #     iaa.EdgeDetect(alpha=(0, 0.7)),
                           #     iaa.DirectedEdgeDetect(
                           #         alpha=(0, 0.7), direction=(0.0, 1.0)
                           #     ),
                           # ])),

                           # 浮雕效果(很奇怪的操作,不确定能不能用)
                           # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # 将1%到10%的像素设置为黑色或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),
                           #
                           # # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                           # sometimes(
                           #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           # ),
                       ],

                       random_order=True  # 随机的顺序把这些操作用在图像上
                       )
        ],
        random_order=True  # 随机的顺序把这些操作用在图像上

    )

    seq_det = seq.to_deterministic()  # 确定一个数据增强的序列
    segmaps = ia.SegmentationMapsOnImage(masks, shape=masks.shape)  # 分割标签格式
    images_aug = seq_det.augment_image(imgs)
    # 将方法应用在分割标签上，并且转换成np类型
    segmaps_aug = seq_det.augment_segmentation_maps(segmaps)
    segmaps_aug = segmaps_aug.get_arr().astype(np.uint8)

    # plt.subplot(2, 2, 1)
    # plt.imshow(np.array(imgs), cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(np.array(masks), cmap='binary_r')
    # plt.subplot(2, 2, 3)
    # plt.imshow(np.array(images_aug), cmap='gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(np.array(segmaps_aug), cmap='binary_r')
    # plt.show()

    return images_aug, segmaps_aug


def data_aug_multimod(imgs_all, masks_all):
    """
    输入3维图像和标签（不够维数的需要在最后面补维）,返回进行了相同扩增的图像和标签,输入输出格式为numpy
    imgs_all的通道数代表了有多少模态；
    masks_all是所有标签类型的数据，其最后一个通道是GT
    """
    # 设定扩增方法
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 设定随机函数
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # 50%图像进行水平翻转
            iaa.Flipud(0.5),  # 50%图像做垂直翻转

            sometimes(iaa.Crop(percent=(0, 0.2))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%

            sometimes(iaa.Affine(  # 对一部分图像做仿射变换
                scale={"x": (0.75, 1.25), "y": (0.75, 1.25)},  # 图像缩放为80%到120%之间
                translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},  # 平移±20%之间
                rotate=(-45, 45),  # 旋转±45度之间
                shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
                order=[0, 1],  # 使用最邻近差值或者双线性差值
                cval=(0, 255),
                mode=ia.ALL,  # 边缘填充,随机0到255间的值
            )),

            # 使用下面的0个到4个之间的方法去增强图像
            iaa.SomeOf((0, 4),
                       [
                           # 锐化处理
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)),
                           # 扭曲图像的局部区域
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # 改变对比度
                           iaa.contrast.LinearContrast((0.6, 1.4), per_channel=0.5),
                           # 将1%到10%的像素设置为黑色或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                           # iaa.OneOf([
                           #     iaa.Dropout((0.01, 0.05), per_channel=0.5),
                           #     iaa.CoarseDropout(
                           #         (0.0, 0.05), size_percent=(0.02, 0.05),
                           #         per_channel=0.2
                           #     ),
                           # ])
                       ],
                       random_order=True  # 随机的顺序把这些操作用在图像上
                       )
        ],
        random_order=True  # 随机的顺序把这些操作用在图像上
    )
    seq_det = seq.to_deterministic()  # 确定一个数据增强的序列

    # ============== 图像增强处理 =====================
    # 对图像的扩增
    mod_num = imgs_all.shape[2]
    for i in range(mod_num):
        # 标准化输入格式
        img = np.array(imgs_all[:, :, i])
        # 线性变化到0-255
        img_max = np.max(img)
        img_min = np.min(img)
        if (img_max - img_min) == 0:    # pet是全0层，ct是全为-360的层，空白层线性变化会导致出现nan
            img_aug = img   # 0就不需要做扩增了
        else:
            img = (img - img_min) / (img_max - img_min) * 255
            # 做扩增
            img_aug = seq_det.augment_image(img)
            # 变化会原来数值范围
            img_aug = img_aug / 255 * (img_max - img_min) + img_min
        # 放回imgs_all
        imgs_all[:, :, i] = img_aug
    # 对label的扩增
    label_num = masks_all.shape[2]
    for i in range(label_num):
        mask = np.array(masks_all[:, :, i])
        # 线性变化到0-255
        mask_min = np.min(mask)
        mask_max = np.max(mask)
        if mask_max - mask_min == 0:  # 无病灶层，防止空白层导致nan
            mask_aug = (mask - mask_min).astype(np.float32)
        else:
            mask = ((mask - mask_min) / (mask_max - mask_min) * 255).astype(np.uint8)
            # 分割标签格式
            segmap = ia.SegmentationMapsOnImage(mask, shape=mask.shape)
            # 将方法应用在分割标签上，并且转换回np类型
            mask_aug = seq_det.augment_segmentation_maps(segmap)
            mask_aug = mask_aug.get_arr()       # 有时会是bool型，大部分时候是uint8，如果是bool则不需要更改范围
            # 原来数值范围
            if mask_aug.dtype == 'bool':
                mask_aug = mask_aug.astype(np.float32)
            else:
                mask_aug = mask_aug / 255 * (mask_max - mask_min) + mask_min
        # 放回masks_all
        masks_all[:, :, i] = mask_aug

    return imgs_all, masks_all
