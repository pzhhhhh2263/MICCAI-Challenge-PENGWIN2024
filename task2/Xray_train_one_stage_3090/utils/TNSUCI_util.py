from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import os
import warnings
import random
import numpy as np
import csv
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings('ignore')
sep = os.sep
filesep = sep  # 设置分隔符


def char_color(s, front=50, word=32):
    """
    # 改变字符串颜色的函数
    :param s:
    :param front:
    :param word:
    :return:
    """
    new_char = "\033[0;"+str(int(word))+";"+str(int(front))+"m"+s+"\033[0m"
    return new_char


def array_shuffle(x, axis=0, random_state=2020):
    """
    对多维度数组，在任意轴打乱顺序
    :param x: ndarray
    :param axis: 打乱的轴
    :return:打乱后的数组
    """
    new_index = list(range(x.shape[axis]))
    random.seed(random_state)
    random.shuffle(new_index)
    x_new = np.transpose(x, ([axis]+[i for i in list(range(len(x.shape))) if i is not axis]))
    x_new = x_new[new_index][:]
    new_dim = list(np.array(range(axis))+1)+[0]+list(np.array(range(len(x.shape)-axis-1))+axis+1)
    x_new = np.transpose(x_new, tuple(new_dim))
    return x_new


def get_filelist_frompath(filepath, expname, sample_id=None):
    """
    读取文件夹中带有固定扩展名的文件
    :param filepath:
    :param expname: 扩展名，如'h5','PNG'
    :param sample_id: 可以只读取固定患者id的图片
    :return: 文件路径list
    """
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.'+expname):
                id = int(file.split('.')[0])        # 以`.`为分隔符,然后第一个,也就得到id
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.'+expname):
                file_List.append(os.path.join(filepath, file))
    return file_List


def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def get_loocv_filelist(csv_file, num_id=1):
    """
    获取留一法结果的API
    :param csv_file: 带有ID、CATE、size的文件
    :param fold: 返回第几折，从1开始
    :return: 指定折的训练集和测试集
    """
    csvlines = readCsv(csv_file)
    header = csvlines[0]
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))

    patient_num = list(set(patient_id))  # 按病人分折

    loo = LeaveOneOut()
    results = []
    for train_index, test_index in loo.split(patient_num):
        if test_index[0]+1 != num_id:
            continue
        train_id = np.array(patient_num)[train_index]
        test_id = np.array(patient_num)[test_index]
        print('train_id:'+str(train_id)+'\nvalid_id:'+str(test_id))

        train_set = []
        test_set = []

        for h5_file in data_id:
            if int(h5_file.split('_')[0]) in test_id:
                test_set.append(h5_file)
            else:
                train_set.append(h5_file)

        results.append([train_set, test_set])

    return results

def get_fold_filelist_pzh(train_id, valid_id, csv_file, K=5, fold=1, random_state=42, validation=False, validation_r=0.2):
    """
   获取分折结果的API（）
   :param csv_file: 带有ID、CATE、size的文件
   :param K: 分折折数
   :param fold: 返回第几折,从1开始
   :param random_state: 随机数种子
   :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
   :param validation_r: 抽取出验证集占训练集的比例
   :return: train和test的h5_list
    """
    csvlines = readCsv(csv_file)
    header = csvlines[0]
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    # print('train_id:' + str(train_id) + '\nval_id:' + str(val_id))

    train_set = []
    val_set = []
    # test_set = []

    for h5_file in data_id:
        if str(h5_file.split('_')[0]) in train_id:
            train_set.append(h5_file)
        elif str(h5_file.split('_')[0]) in valid_id:
            val_set.append(h5_file)
        # else:
        #     test_set.append(h5_file)

    return train_set, val_set


def get_fold_filelist_sn(csv_file, K=3, fold=1, random_state=2020, validation=False, validation_r = 0.2):
    """
       获取分折结果的API（）
       :param csv_file: 带有ID、CATE、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param random_state: 随机数种子
       :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
       :param validation_r: 抽取出验证集占训练集的比例
       :return: train和test的h5_list
       """
    csvlines = readCsv(csv_file)
    header = csvlines[0]
    # print('header', header)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]
    
    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))

    patient_num = list(set(patient_id))     # 按病人分折

    fold_train = []
    fold_test = []
    
    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(patient_num):
        fold_train.append(np.array(patient_num)[train_index])
        fold_test.append(np.array(patient_num)[test_index])

    train_id = fold_train[fold-1]
    test_id = fold_test[fold-1]
    print('train_id:'+str(train_id)+'\nvalid_id:'+str(test_id))

    train_set = []
    test_set = []

    for h5_file in data_id:
        if int(h5_file.split('_')[0]) in test_id:
            test_set.append(h5_file)
        else:
            train_set.append(h5_file)
    return [train_set, test_set]

def get_fold_filelist_pzh(train_id, valid_id, csv_file, K=5, fold=1, random_state=42, validation=False, validation_r=0.2):
    """
   获取分折结果的API（）
   :param csv_file: 带有ID、CATE、size的文件
   :param K: 分折折数
   :param fold: 返回第几折,从1开始
   :param random_state: 随机数种子
   :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
   :param validation_r: 抽取出验证集占训练集的比例
   :return: train和test的h5_list
    """
    csvlines = readCsv(csv_file)
    header = csvlines[0]
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    # print('train_id:' + str(train_id) + '\nval_id:' + str(val_id))

    train_set = []
    val_set = []
    # test_set = []

    for h5_file in data_id:
        if str(h5_file.split('_')[0]) in train_id:
            train_set.append(h5_file)
        elif str(h5_file.split('_')[0]) in valid_id:
            val_set.append(h5_file)
        # else:
        #     test_set.append(h5_file)

    return train_set, val_set

def get_fold_filelist_train_some(csv_file, K=5, fold=1, extract_num=1, random_state=2020):
    """
       获取训练集里的设定例数的全身h5_list
       :param csv_file: 带有ID、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param extract_num: 抽取训练集内几例的全身图像
       :param random_state: 随机数种子
       :return: train的h5_list
    """
    csvlines = readCsv(csv_file)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))
    patient_num = list(set(patient_id))     # 按病人分折

    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    fold_train = []
    for train_index, test_index in kf.split(patient_num):
        fold_train.append(np.array(patient_num)[train_index])

    train_id = fold_train[fold-1]
    # 从训练集内随机抽取extract_num个id
    np.random.seed(random_state)        # 保证可重复性
    extract_train_id = np.random.choice(train_id, extract_num, replace=False)
    print('whole_body train_id:'+str(extract_train_id))

    train_set = []
    for h5_file in data_id:
        if int(h5_file.split('_')[0]) in extract_train_id:
            train_set.append(h5_file)
    return train_set


def get_fold_filelist_train_all(csv_file, K=5, fold=1, extract_num=1, random_state=2020):
    """
       在训练集里按设定例数划分，取全身h5_list，组成一个集合
       :param csv_file: 带有ID、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param extract_num: 抽取训练集内几例的全身图像
       :param random_state: 随机数种子
       :return: train的h5_list的集合
    """
    csvlines = readCsv(csv_file)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))
    patient_num = list(set(patient_id))     # 按病人分折

    fold_train = []
    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(patient_num):
        fold_train.append(np.array(patient_num)[train_index])

    train_id = fold_train[fold-1]
    train_num_limit = int(len(train_id)/extract_num)*extract_num       # 根据取的数目，限制训练集取的范围，即drop_last
    train_set_all = []
    for i in range(0, train_num_limit, extract_num):
        # 训练集内按间隔取
        extract_train_id = train_id[i:i+extract_num]
        train_set = []
        for h5_file in data_id:
            if int(h5_file.split('_')[0]) in extract_train_id:
                train_set.append(h5_file)
        train_set_all.append(train_set)
    return train_set_all


def get_fold_filelist_train_all_3d(csv_file, K=20, fold=0, extract_num=16, random_state=2020):
    """
       在训练集里按设定例数划分，取全身h5_list，组成一个集合
       :param csv_file: 带有ID、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param extract_num: 抽取训练集内几例的全身图像
       :param random_state: 随机数种子
       :return: train的h5_list的集合
    """
    csvlines = readCsv(csv_file)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))
    patient_num = list(set(patient_id))     # 按病人分折

    fold_train = []
    fold_Extrain = []

    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(patient_num):
        fold_train.append(np.array(patient_num)[train_index])
        fold_Extrain.append(np.array(patient_num)[test_index])
    
    train_set_Ex = []

    for h5_file in data_id:
        if int(h5_file.split('_')[0]) in fold_Extrain[fold]:
            train_set_Ex.append(h5_file)

    return train_set_Ex


def get_fold_filelist_4all(csv_file, K=5, fold=1, extract_num=1, random_state=2020, lap_rate=0.25):
    """
        在全身h5文件中获取
        :param extract_num: 从全身的数据中取多少个病人
        :param csv_file: 所有文件的CSV呗
        :param lap_rate: 重叠比率
        返回一个装满分折病人的每一切片文件名的列表
    """
    csvlines = readCsv(csv_file)
    file_data = csvlines[1:]
    file_path = [i[0] for i in file_data]
    patient_num = [int(path.split('_')[0]) for path in file_path]
    patient_set = list(set(patient_num))
    train_fold = []
    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(patient_set):  # 获取k分割的索引值
        train_fold.append(np.array(patient_set)[train_index])

    train_id = train_fold[fold - 1]
    lap = int(extract_num * lap_rate)  # 每个set重叠的数量
    block_num = len(train_id)  #所有train的数量

    train_num_limit = int(block_num / (extract_num - lap)) * (extract_num - lap)  # 限制取训练集的数量，保证均匀放置这些病人(不多拿不少拿)
    train_set_all = []
    # 全新（彩色）数据排列
    new_patient_list = []
    for i in range(0, train_num_limit, extract_num - lap):
        extra_list = train_id[i:i + extract_num - lap]
        new_patient_list.append(extra_list)
    num_round = len(new_patient_list)
    whole_body_set = []
    whole_body_set.append(new_patient_list[0])
    # 后几排的排列
    for i in range(1, num_round):
        last_set = whole_body_set[i - 1]
        lap_list = np.random.choice(last_set, lap, replace=False)  # 重复用的那部分的病人     # np array
        extra_list = new_patient_list[i]  # train id 也是np array
        extra_list = np.concatenate([extra_list, lap_list])
        whole_body_set.append(extra_list)
    # 第一排和最后一排的重叠
    last_set = whole_body_set[-1]
    lap_list = np.random.choice(last_set, lap, replace=False)
    extra_list = new_patient_list[0]
    extra_list = np.concatenate([extra_list, lap_list])
    whole_body_set[0] = extra_list  # 在最开始插入第一行的排列

    for ex_list in whole_body_set:
        train_set = []
        for h5_file in file_path:
            if int(h5_file.split("_")[0]) in ex_list:
                train_set.append(h5_file)
        train_set_all.append(train_set)
    return train_set_all

def get_wholebody_test_randomfilelist(csv_file, K=5, fold=1, random_state=2020, random_rate=0.3):
    """
    为了在训练中得到全身验证的分数，随机采样病人提高效率
    :param csv_file:
    :param K: fold
    :param fold: fold
    :param:random_rate: 随机多少个数
    :return: 随机采样的set和random或不random的id给之后的全身测试用
    """
    csvlines = readCsv(csv_file)
    file_data = csvlines[1:]
    file_path = [i[0] for i in file_data]
    patient_num = [int(path.split('_')[0]) for path in file_path]
    patient_set = list(set(patient_num))  # 按病人分折  set:不重复的集合
    val_fold = []
    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)  # 分K折
    for _, test_index in kf.split(patient_set):  # 获取k分割的索引值
        val_fold.append(np.array(patient_set)[test_index])
    test_id = val_fold[fold - 1]
    if random_rate is not None:
        random_id = np.random.choice(test_id, int(random_rate * len(test_id)), replace=False).tolist()  # 随机选random_num个
    else:
        random_id = test_id  # 不随机选，全部验证
    test_set = []
    # 取那些test_id内的图片
    for path in file_path:
        if int(path.split("_")[0]) in random_id:
            test_set.append(path)
    return test_set, random_id

def get_fold_filelist_train_all_v2(csv_file, K=5, fold=1, extract_num=1, ratio=0.25, random_state=2020):
    """
       在训练集里按设定例数划分，取全身h5_list，组成一个集合,v2版本：list前后有交叉
       :param csv_file: 带有ID、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param extract_num: 抽取训练集内几例的全身图像
       :param ratio: 前后list的数据交叉程度
       :param random_state: 随机数种子
       :return: train的h5_list的集合
    """
    csvlines = readCsv(csv_file)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))
    patient_num = list(set(patient_id))     # 按病人分折

    fold_train = []

    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(patient_num):
        fold_train.append(np.array(patient_num)[train_index])

    train_id = fold_train[fold-1]
    train_num_limit = int(len(train_id)/extract_num)*extract_num       # 根据取的数目，限制训练集取的范围，即drop_last
    train_set_all = []
    over_extract_num = int(extract_num * ratio)
    for i in range(0, train_num_limit, over_extract_num):
        # 训练集内按间隔取
        extract_train_id = train_id[i:i+extract_num]
        train_set = []
        for h5_file in data_id:
            if int(h5_file.split('_')[0]) in extract_train_id:
                train_set.append(h5_file)
        train_set_all.append(train_set)
    return train_set_all


def save_nii(save_nii, CT_nii, save_path, save_mask=True):
    """
    保存nii
    :param save_nii: 需要保存的nii图像的array
    :param CT_nii: 配准的图像，用于获取同样的信息
    :param save_path: 保存路径
    :param save_mask: 保存的是否是mask，默认是True
    :return:
    """
    if save_mask:
        # 保存mask_nii
        save_sitk = sitk.GetImageFromArray(save_nii.astype(np.uint8))
        save_sitk.CopyInformation(CT_nii)
        save_sitk = sitk.Cast(save_sitk, sitk.sitkUInt8)
    else:
        # 保存img_nii
        save_sitk = sitk.GetImageFromArray(save_nii.astype(np.float))
        save_sitk.CopyInformation(CT_nii)
        save_sitk = sitk.Cast(save_sitk, sitk.sitkFloat32)

    sitk.WriteImage(save_sitk, save_path)
    print(save_path+' processing successfully!')


def print_logger(logger, savepth):
    for index, key in enumerate(logger.keys()):
        figg = plt.figure()
        plt.plot(logger[key])
        figg.savefig(savepth + sep + key + '.PNG')
        plt.close()


def center_crop(imgs, body_mask, new_size):
    labeled_img, _ = measure.label(body_mask, connectivity=3, return_num=True)
    body_centroid = measure.regionprops(labeled_img)[0].centroid  # z,y,x
    img_crops = []
    for img in imgs:
        img_crop = img[:, int(body_centroid[1]-new_size/2):int(body_centroid[1]-new_size/2)+new_size,
                   int(body_centroid[2]-new_size/2):int(body_centroid[2]-new_size/2)+new_size]
        img_crops.append(img_crop)
    return img_crops


if __name__ == '__main__':

    all_csv_file = r'/home/user4/sharedata/newnas_1/PZH/LCa/csv_data/0-399_crop320_all.csv'
    fold_K = 20
    fold_idx = 0
    extra_num = 16
    seed = 42

    extra_train_set = get_fold_filelist_train_all_3d(csv_file = all_csv_file, K=fold_K, fold=fold_idx,
                                                  extract_num = extra_num, random_state=seed)
    print(extra_train_set)