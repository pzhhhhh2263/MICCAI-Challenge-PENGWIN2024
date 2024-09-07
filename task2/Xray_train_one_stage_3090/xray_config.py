import argparse
parser = argparse.ArgumentParser()
# =============================常改的参数=============================
parser.add_argument('--Task_name', type=str, default='only1stage_Xray_hip_right_ch10_numall_zscore_MAnet_pos_weight+focal')  # 任务名,也是文件名
parser.add_argument('--cuda_idx', type=int, default=0)  # 用几号卡的显存
parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')  # 交叉验证的折数
parser.add_argument('--fold_idx', type=int, default=0)  # 跑第几折的数据
parser.add_argument('--fold_all', type=str, default='all', help='folds number after divided')  # 是否全部数据都训练

parser.add_argument('--leave_one', default=False)  # 是否留一验证
parser.add_argument('--num_id', default=2)  # 留哪个病人

parser.add_argument('--batch_size', type=int, default=8)  # 训多少个图才回调参数
parser.add_argument('--batch_size_test', type=int, default=2)  # 测试时多少个,可以设很大,但结果图就会很小
parser.add_argument('--num_workers', type=int, default=8)


# =============================偶尔改的参数=============================
# data-parameters
# parser.add_argument('--filepath_img', type=str, default=r'/data/newnas_1/PZH/LCa/h5_data/512_all')
# parser.add_argument('--pretrained_path', type=str,
#                     default=r'/home/user4/sharedata/newnas_1/PZH/code/My_Model_try/model_pretrained/R50+ViT-B_16.npz')
# parser.add_argument('--filepath_img', type=str, default=r'/home/user4/sharedata/nas/heyixue_group/LCa/h5_data_pre/320')

# parser.add_argument('--filepath_img', type=str, default=r'C:\Users\siat\PycharmProjects\xray_seg_code_2d\seg_code_2d\Xray_train_v2\hip_right_10channels')
# parser.add_argument('--filepath_img', type=str, default=r'/home/siat/pzh/pengwin/two_stage/h5_data/hip_right') # h5数据集地址
parser.add_argument('--filepath_img', type=str, default=r'/home/siat/pzh/pengwin/only1_stage') # npz数据集地址

# 用于分折的csv表格
parser.add_argument('--csv_file', type=str, default=r'/home/siat/pzh/pengwin/two_stage/csv_data/file_names_npz.csv')
# parser.add_argument('--csv_file', type=str, default=r'/home/siat/pzh/pengwin/two_stage/csv_data/file_names_num1.csv')
# parser.add_argument('--csv_file', type=str, default=r'/home/siat/pzh/pengwin/two_stage/csv_data/file_names_num100.csv')

# result&save
parser.add_argument('--result_path', type=str, default=r'/home/siat/pzh/pengwin/two_stage/only1_stage/hip_right_results')  # 结果保存地址

parser.add_argument('--dataset', type=str, default='hip_right')  # 结果保存地址


# model hyper-parameters
parser.add_argument('--image_size', type=int, default=448)
parser.add_argument('--monai_model_name', type=str, default=None)  # 模型框架
parser.add_argument('--model_name', type=str, default='smp.MAnet')  # 模型框架
parser.add_argument('--encoder_name', type=str, default='timm-regnety_160')  # 编码结构

# parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')  # 训练权重

# 设置学习率(chrome有问题,额外记录在一个txt里面)
# 注意,学习率记录部分代码也要更改
parser.add_argument('--lr', type=float, default=1e-4)  # 初始or最大学习率(单用lovz且多gpu的时候,lr貌似要大一些才可收敛)
parser.add_argument('--lr_low', type=float, default=1e-6)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)
parser.add_argument('--num_epochs', type=int, default=40)  # 总epoch
parser.add_argument('--lr_cos_epoch', type=int, default=40)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用
parser.add_argument('--lr_warm_epoch', type=int, default=0)  # warm_up的epoch数,一般就是10~20,为0或False则不使用
parser.add_argument('--save_model_step', type=int, default=10)  # 多少epoch保存一次模型
parser.add_argument('--val_step', type=int, default=2)  #多少epoch验证一次

# =============================一般不改的参数=============================
parser.add_argument('--mode', type=str, default='train', help='train/test')  # 训练还是测试
parser.add_argument('--num_epochs_decay', type=int, default=2)  # decay开始的最小epoch数
parser.add_argument('--decay_ratio', type=float, default=0.1)  # 0~1,每次decay到1*ratio
parser.add_argument('--decay_step', type=int, default=80)  # 暂时没用

# optimizer param
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
# parser.add_argument('--augmentation_prob', type=float, default=0.3)  # 数据扩增的概率
parser.add_argument('--augmentation_prob', type=float, default=0)  # 数据扩增的概率

parser.add_argument('--save_detail_result', type=bool, default=True)
parser.add_argument('--save_image', type=bool, default=True)  # 训练过程中观察图像和结果
# training hyper-parameters
parser.add_argument('--img_ch', type=int, default=1)
# parser.add_argument('--output_ch', type=int, default=2)
parser.add_argument('--output_ch', type=int, default=10)
parser.add_argument('--DataParallel', type=bool, default=False)  # 数据并行,开了可以用多张卡的显存,不推荐使用
parser.add_argument('--train_flag', type=bool, default=False)  # 训练过程中是否测试训练集,不测试会节省很多时间
parser.add_argument('--seed', type=int, default=42)  # 随机数的种子点，一般不变
parser.add_argument('--TTA', type=bool, default=False)

config = parser.parse_args()
