"""
X光2D训练主程序
"""
# import sys
# sys.path.append(r'C:\Users\siat\PycharmProjects\xray_seg_code_2d/seg_code_2d')

import datetime
import torch
from torch.backends import cudnn
import json
from data_loader_xray import get_loader_2d_xray_npz
from xray_solver import Solver
from utils.TNSUCI_util import *
from xray_config import config      # 参数设置在这里


if __name__ == '__main__':
    # step1: 设置随机数种子 -------------------------------------------------------
    seed = config.seed
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # step2: 设置各种文件保存路径 -------------------------------------------------------
    # 结果保存地址，后缀加上fold
    config.result_path = os.path.join(config.result_path,
                                      config.Task_name + '_fold' + str(config.fold_K) + '-' + str(config.fold_idx))
    config.model_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')       # 在终端用 tensorboard --logdir=地址 指令查看指标
    config.log_pic_dir = os.path.join(config.result_path, 'logger_pic')
    config.writer_4SaveAsPic = dict(lr=[], loss=[], loss_DICE=[], loss_BCE=[], loss_LOVAZ=[], score_val=[])
    # Create directories if not exist
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        os.makedirs(config.model_path)
        os.makedirs(config.log_dir)
        os.makedirs(config.log_pic_dir)
        os.makedirs(os.path.join(config.result_path, 'images'))
    config.record_file = os.path.join(config.result_path, 'record.txt')
    f = open(config.record_file, 'a')
    f.close()

    # 保存设置到txt
    print(config)
    f = open(os.path.join(config.result_path, 'config.txt'), 'w')
    for key in config.__dict__:
        print('%s: %s' % (key, config.__getattribute__(key)), file=f)
    f.close()

    # step3: GPU device -------------------------------------------------------
    cudnn.benchmark = True
    if not config.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)

    # step4: 数据集获取 -------------------------------------------------------
    # # 改写了分折方式,确保按病人分折
    if config.leave_one:
        results =  get_loocv_filelist(config.csv_file, num_id=config.num_id)
        train_set = results[0][0]
        test_set = results[0][1]
        train_list = [config.filepath_img + sep + i for i in train_set]
        valid_list = [config.filepath_img + sep + i for i in test_set]
        config.train_list = train_list
        config.valid_list = valid_list


    elif config.fold_all != 'all':
        train, valid = get_fold_filelist_sn(config.csv_file, K=config.fold_K, fold=config.fold_idx, random_state=seed)
        train_list = [config.filepath_img + sep + i for i in train]
        valid_list = [config.filepath_img + sep + i for i in valid]

        config.train_list = train_list
        config.valid_list = valid_list

    elif config.fold_all == 'all':
        train, valid = get_fold_filelist_sn(config.csv_file, K=config.fold_K, fold=config.fold_idx, random_state=seed)
        train_list = [config.filepath_img + sep + i for i in train]
        valid_list = [config.filepath_img + sep + i for i in valid]
        all_list = train_list + valid_list
        config.train_list = all_list
        config.valid_list = all_list[::500]
        # 新列表将包含100个元素
        print(len(config.valid_list))  # 输出100
        print(config.valid_list)



    # 读取h5文件，交叉验证，只有训练和验证
    train_loader = get_loader_2d_xray_npz(
        h5_list=config.train_list,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset=config.dataset,
        mode='train',
        augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader_2d_xray_npz(
        h5_list=config.valid_list,
        image_size=config.image_size,
        batch_size=config.batch_size_test,
        num_workers=config.num_workers,
        dataset=config.dataset,
        mode='valid',
        augmentation_prob=0.0)

    # step4: 网络设置（包括学习率方法）、模型训练测试方案 -------------------------------------
    solver = Solver(config, train_loader, valid_loader)

    # step5: 训练or测试 -------------------------------------------------------
    if config.mode == 'train':
        print('分折：%d-%d' % (config.fold_K, config.fold_idx))
        solver.train()
    elif config.mode == 'test':         # 这里的测试本质上是单病灶区的2d验证，真正的验证和测试需要全身测试，额外写代码实现
        tic = datetime.datetime.now()  # 计时
        unet_best_path = os.path.join(config.model_path, 'best.pkl')
        print('=================================== test(val) ===================================')
        acc, SE, SP, PC, DC, IOU = solver.test(mode='val', unet_path=unet_best_path)
        print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
            acc, SE, SP, PC, DC, IOU))

        toc = datetime.datetime.now()
        h, remainder = divmod((toc - tic).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "per epoch testing&vlidation cost Time %02d h:%02d m:%02d s" % (h, m, s)
        print(char_color(time_str))

