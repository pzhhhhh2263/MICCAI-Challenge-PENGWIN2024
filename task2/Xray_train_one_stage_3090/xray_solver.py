import datetime
import os
import time
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
import ttach as tta
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from loss.dice_loss import FocalTversky_loss
from loss.loss_weight import *
from loss.lovasz_losses import lovasz_hinge, binary_xloss
from utils.TNSUCI_util import *
from utils.scheduler import GradualWarmupScheduler
from utils.evaluation import *
from utils.misc import printProgressBar
from monai.networks.nets import SegResNet
# from My_Model_v2.modeling_irene import VisionTransformer as Vit_Seg
# from My_Model_v2.modeling_irene import CONFIGS as CONFIGS_ViT_seg


class Solver(object):
    def __init__(self, config, train_loader, valid_loader):
    # def __init__(self, config,valid_loader):
        # Make record file
        self.record_file = config.record_file

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device.type+':'+os.environ["CUDA_VISIBLE_DEVICES"])

        self.Task_name = config.Task_name

        # Data loader
        self.num_workers = config.num_workers
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Models
        self.monai_model_name = config.monai_model_name
        self.unet = None  # 模型，基本确定是使用unet结构
        self.model_name = config.model_name
        self.encoder_name = config.encoder_name
        # self.vit_name = CONFIGS_ViT_seg[config.vit_name]
        # self.vit_name.n_classes = config.num_classes
        # self.pretrained_path = config.pretrained_path

        self.optimizer = None
        self.img_ch = config.img_ch
        self.image_size = config.image_size
        self.output_ch = config.output_ch
        self.augmentation_prob = config.augmentation_prob
        self.img_size = config.image_size
        # loss
        self.criterion = lovasz_hinge
        # self.criterion1 = binary_xloss
        pos_weights = torch.tensor([3.0] * 2 + [1.0] * 8).to(self.device)  # 对应于每个通道的权重
        self.criterion1 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        # self.criterion2 = SoftDiceLoss()
        self.criterion2 = SoftDiceLoss2()
        # self.criterion2 = SoftDiceLoss3()

        self.criterion3 = torch.nn.CrossEntropyLoss()
        self.criterion4 = FocalTversky_loss(alpha=0.7,beta=0.3)

        # self.criterion4 = FocalTversky_loss()
        self.lw = AutomaticWeightedLoss(device=self.device, num=3)

        # Hyper-parameters
        self.lr = config.lr
        self.lr_low = config.lr_low
        if self.lr_low is None:
            self.lr_low = self.lr / 1e+6
            print("auto set minimun lr :", self.lr_low)

        # optimizer param
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.batch_size_test = config.batch_size_test

        # Step size
        self.save_model_step = config.save_model_step
        self.val_step = config.val_step
        self.decay_step = config.decay_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.save_image = config.save_image
        self.save_detail_result = config.save_detail_result
        self.log_dir = config.log_dir
        self.writer_4SaveAsPic = config.writer_4SaveAsPic
        self.log_pic_dir = config.log_pic_dir

        # 设置学习率策略相关参数
        self.decay_ratio = config.decay_ratio
        self.num_epochs_decay = config.num_epochs_decay
        self.lr_cos_epoch = config.lr_cos_epoch
        self.lr_warm_epoch = config.lr_warm_epoch
        self.lr_sch = None  # 初始化先设置为None
        self.lr_list = []  # 临时记录lr

        # 其他参数
        self.DataParallel = config.DataParallel
        self.train_flag = config.train_flag
        self.TTA = config.TTA
        if self.TTA:
            print('use TTA')  # 测试时扩增,一种提升结果的trick

        # 执行个初始化函数
        self.my_init()

    def myprint(self, *args):
        """Print & Record while training."""
        print(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()

    def my_init(self):
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        self.print_date_msg()
        self.build_model()

    def print_date_msg(self):
        self.myprint("images count in train:{}".format(len(self.train_loader.dataset)))
        self.myprint("images count in valid:{}".format(len(self.valid_loader.dataset)))

    def build_model(self):  # 构建网络,记得修改!
        """Build generator and discriminator."""
        # 用monai构建网络
        if self.monai_model_name != None:
            self.unet = SegResNet(
                spatial_dims=2,
                init_filters=32,
                in_channels=self.img_ch,
                out_channels=self.output_ch,  
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1]
            ) 
            print("Bulid model with " + self.monai_model_name)
        else:
        # 用smp构建网络

            self.unet = eval(self.model_name)(encoder_name=self.encoder_name,
                                        encoder_weights='imagenet',
                                        in_channels=self.img_ch, classes=self.output_ch)

            print("Bulid model with " + self.model_name + ',encoder:' + self.encoder_name + ' version:' + smp.__version__)

        # 用smp构建网络
        # self.unet = Vit_Seg(self.vit_name, img_size=self.image_size, num_classes=self.vit_name.n_classes).cuda()
        # self.unet.load_from(weights=np.load(self.pretrained_path))
        # self.unet = AttU_Net(in_ch=self.img_ch, out_ch=self.output_ch)
        # print("Bulid model with AttU_Net")

        # 优化器修改
        self.optimizer = optim.AdamW(list(self.unet.parameters()), self.lr, (self.beta1, self.beta2))

        # lr schachle策略(要传入optimizer才可以)	学习率下降策略
        # 暂时的三种情况,(1)只用cos余弦下降,(2)只用warmup预热,(3)两者都用
        if self.lr_warm_epoch != 0 and self.lr_cos_epoch == 0:  # 只用预热
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr,然后在一定epoch内升回初始学习率lr
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=None)
            print('use warmup lr sch')
        elif self.lr_warm_epoch == 0 and self.lr_cos_epoch != 0:  # 只用余弦下降,在lr_cos_epoch内下降到最低学习率lr_low
            self.lr_sch = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                         self.lr_cos_epoch,
                                                         eta_min=self.lr_low)
            print('use cos lr sch')
        elif self.lr_warm_epoch != 0 and self.lr_cos_epoch != 0:  # 都用
            self.update_lr(self.lr_low)
            scheduler_cos = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                           self.lr_cos_epoch,
                                                           eta_min=self.lr_low)
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=scheduler_cos)
            print('use warmup and cos lr sch')
        else:
            if self.lr_sch is None:
                print('use decay coded by dasheng')

        self.unet.to(self.device)
        if self.DataParallel:
            self.unet = torch.nn.DataParallel(self.unet)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.myprint(model)
        self.myprint(name)
        self.myprint("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable from tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, lr):
        """Update the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def tensor2img(self, x):
        """Convert tensor to img (numpy)."""
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""
        self.myprint('-----------------------%s-----------------------------' % self.Task_name)
        # 最优以及最新模型保存地址
        unet_best_path = os.path.join(self.model_path, 'best.pkl')
        unet_lastest_path = os.path.join(self.model_path, 'lastest.pkl')
        writer = SummaryWriter(log_dir=self.log_dir)

        # 判断是不是被中断的，如果是，那么就重载断点继续训练
        # 重新加载的参数包括：
        # 参数部分 1）模型权重；2）optimizer的参数，比如动量之类的；3）schedule的参数；4）epoch
        if os.path.isfile(unet_lastest_path):
            self.myprint('Reloading checkpoint information...')
            latest_status = torch.load(unet_lastest_path)
            self.unet.load_state_dict(latest_status['model'])
            self.optimizer.load_state_dict(latest_status['optimizer'])
            self.lr_sch.load_state_dict(latest_status['lr_scheduler'])
            self.writer_4SaveAsPic = latest_status['writer_4SaveAsPic']
            print('restart at epoch:', latest_status['epoch'])
            best_unet_score = latest_status['best_unet_score']
            best_epoch = latest_status['best_epoch']
            epoch_start = latest_status['epoch']
            Iter = latest_status['Iter']
        else:
            best_unet_score = 0.0
            best_epoch = 1
            epoch_start = 0
            Iter = 0

        valid_record = np.zeros((1, 8))  # [epoch, Iter, acc, SE, SP, PC, Dice, IOU]     # 记录指标

        self.myprint('Training...')
        for epoch in range(epoch_start, self.num_epochs):
            tic = datetime.datetime.now()
            self.unet.train(True)
            epoch_loss = 0
            length = 0

            train_len = len(self.train_loader)

            for i, sample in enumerate(self.train_loader):
                (_, images, GT) = sample
                images = images.to(self.device)
                GT = GT.to(self.device)
                # 计算loss
                # SR : Segmentation Result
                SR = self.unet(images)
                # 3,448,448 01
                # print('max logits = ',torch.max(SR))  # 查看是否真的是logits
                # SR_probs = torch.softmax(SR, dim = 1)
                SR_probs = F.sigmoid(SR)

                SR_flat = SR_probs.view(SR_probs.size(0), -1)
                GT_flat = GT.view(GT.size(0), -1)
                # SR_logits_sq = torch.squeeze(SR)
                # GT_sqz = torch.squeeze(GT)
                # print(SR_logits_sq.shape)
                # print(GT_sqz.shape)

                # 计算loss
                loss_soft_dice = self.criterion2(SR_flat, GT_flat)
                # loss_lovz = self.criterion(SR_logits_sq, GT_sqz)
                # loss_ce = self.criterion3(SR, GT)

                # loss_bi_BCE = self.criterion1(SR, GT)
                # loss_focal = self.criterion4(SR_flat, GT_flat)
                # 重塑 output 和 target 张量
                SR_ch = SR.permute(0, 2, 3, 1).reshape(-1, 10)
                GT_ch = GT.permute(0, 2, 3, 1).reshape(-1, 10)

                loss_bi_BCE = self.criterion1(SR_ch, GT_ch)
                loss_focal = self.criterion4(SR_flat, GT_flat)
                # 总loss
                # 设置下各个loss的权重
                lovz_w = 1.0
                soft_dice_w = 1.0
                bi_BCE_w = 1.0
                focal_w = 1.0
                # loss = lovz_w * loss_lovz + soft_dice_w * loss_sofdice + bi_BCE_w * loss_bi_BCE
                # loss = loss_sofdice + loss_bi_BCE
                loss = soft_dice_w * loss_soft_dice + bi_BCE_w * loss_bi_BCE + focal_w * loss_focal

                # 自动学习权重，这个本质上还是得确定大致量级后才能有效，目前感觉和全等差不多
                # loss = self.lw(loss_lovz, loss_sofdice, loss_bi_BCE)

                epoch_loss += float(loss)

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                length += 1
                Iter += 1
                writer.add_scalars('Loss', {'loss': loss}, Iter)

                if self.save_image and (i % 20 == 0):  # 20张图后保存一次png结果
                    images_all = torch.cat((images[:, 0:1, :, :], SR_probs[:, 0:1, :, :], SR_probs[:, 1:2, :, :],
                                            # SR_probs[:, 4:5, :, :],SR_probs[:, 5:6, :, :],SR_probs[:, 6:7, :, :],SR_probs[:, 7:8, :, :],SR_probs[:, 8:9, :, :],SR_probs[:, 9:10, :, :],
                                            GT[:, 0:1, :, :],GT[:, 1:2, :, :]
                                            # GT[:, 4:5, :, :],GT[:, 5:6, :, :],GT[:, 6:7, :, :],GT[:, 7:8, :, :],GT[:, 8:9, :, :],GT[:, 9:10, :, :]
                                            ), 0)
                    torchvision.utils.save_image(images_all.data.cpu(),
                                                 os.path.join(self.result_path, 'images', 'Train_%d_image.png' % i),
                                                 nrow=self.batch_size)

                # 储存loss到list并打印为图片
                current_lr = self.optimizer.param_groups[0]['lr']
                # self.writer_4SaveAsPic['loss'].append(loss.data.cpu().numpy())
                # self.writer_4SaveAsPic['loss_LOVAZ'].append(loss_lovz.data.cpu().numpy())
                self.writer_4SaveAsPic['loss_DICE'].append(loss_soft_dice.data.cpu().numpy())
                self.writer_4SaveAsPic['loss_BCE'].append(loss_bi_BCE.data.cpu().numpy())
                # self.writer_4SaveAsPic['lr'].append(current_lr)
                print_logger(self.writer_4SaveAsPic, self.log_pic_dir)

                # trainning bar
                # print_content = 'batch_total_loss:' + str(loss.data.cpu().numpy()) + \
                #                 '  BCE:' + str(loss_bi_BCE.data.cpu().numpy()) + \
                #                 '  dice:' + str(loss_soft_dice.data.cpu().numpy()) 
                #                 # '  lovasz:' + str(loss_lovz.data.cpu().numpy())
                print_content = 'batch_total_loss: ' + str(round(loss.data.cpu().numpy().item(), 4)) + \
                '  BCE: ' + str(round(loss_bi_BCE.data.cpu().numpy().item(), 4)) + \
                '  dice: ' + str(round(loss_soft_dice.data.cpu().numpy().item(), 4)) + \
                '  focal: ' + str(round(loss_focal.data.cpu().numpy().item(), 4))

                # '  lovasz: ' + str(round(loss_lovz.data.cpu().numpy().item(), 4))

                printProgressBar(i + 1, train_len, content=print_content)

            # 计时结束
            toc = datetime.datetime.now()
            h, remainder = divmod((toc - tic).seconds, 3600)  # 小时和余数
            m, s = divmod(remainder, 60)  # 分钟和秒
            time_str = "per epoch training cost Time %02d h:%02d m:%02d s" % (h, m, s)
            print(char_color(time_str))

            tic = datetime.datetime.now()

            epoch_loss = epoch_loss / length
            self.myprint('Epoch [%d/%d], Loss: %.4f lr: %.8f' % (epoch + 1, self.num_epochs, epoch_loss, current_lr))

            # 记录下lr到log里(并且记录到图片里)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_list.append(current_lr)
            writer.add_scalars('Learning rate', {'lr': current_lr}, epoch)

            self.writer_4SaveAsPic['loss'].append(epoch_loss)
            self.writer_4SaveAsPic['lr'].append(current_lr)
            print_logger(self.writer_4SaveAsPic, self.log_pic_dir)

            # 每个epoch保存lr为png
            figg = plt.figure()
            plt.plot(self.lr_list)
            figg.savefig(os.path.join(self.result_path, 'lr.PNG'))
            plt.close()
            figg, axis = plt.subplots()
            plt.plot(self.lr_list)
            axis.set_yscale("log")
            figg.savefig(os.path.join(self.result_path, 'lr_log.PNG'))
            plt.close()

            # ========================= 学习率策略部分 =========================
            # lr scha way 1:
            # 用上面定义的下降方式
            if self.lr_sch is not None:
                # print("here!!!!!!!!!!")
                if (epoch + 1) <= (self.lr_cos_epoch + self.lr_warm_epoch):
                    self.lr_sch.step()
            # lr scha way 2: Decay learning rate(如果使用方式1,则不使用此方式)
            # 超过num_epochs_decay后,每超过num_epochs_decay后阶梯下降一次lr
            if self.lr_sch is None:
                if ((epoch + 1) >= self.num_epochs_decay) and (
                        (epoch + 1 - self.num_epochs_decay) % self.decay_step == 0):
                    if current_lr >= self.lr_low:
                        self.lr = current_lr * self.decay_ratio
                        self.update_lr(self.lr)
                        self.myprint('Decay learning rate to lr: {}.'.format(self.lr))

            #  ========================= 验证 ===========================
            if (epoch + 1) % self.val_step == 0:
                if self.train_flag:
                    # Train
                    acc, SE, SP, PC, DC, IOU = self.test(mode='train')
                    self.myprint('[Train] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
                        acc, SE, SP, PC, DC, IOU))

                # Validation
                acc, SE, SP, PC, DC, IOU = self.test(mode='valid')
                valid_record = np.vstack((valid_record, np.array([epoch + 1, Iter, acc, SE, SP, PC, DC, IOU])))

                # TODO,以dsc作为最优指标
                unet_score = DC

                # 储存到tensorboard，并打印txt
                writer.add_scalars('Valid', {'Dice': DC, 'IOU': IOU}, epoch)
                self.myprint('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
                    acc, SE, SP, PC, DC, IOU))

                # 储存指标到pix
                self.writer_4SaveAsPic['score_val'].append([DC, IOU])
                print_logger(self.writer_4SaveAsPic, self.log_pic_dir)

                # 保存断点模型和其他参数，用于断点重训，每10个epoch保存一次
                # 发现断点重训的学习率不变，原因是用了预热策略，只用cos则不会
                if (epoch + 1) % 10 == 0:
                    lastest_state = dict(
                        model=self.unet.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        lr_scheduler=self.lr_sch.state_dict(),
                        epoch=epoch+1,      # 此时学习率已经是改变过的，因此重载的epoch应该从下一个开始
                        Iter=Iter,
                        best_epoch=best_epoch,
                        best_unet_score=best_unet_score,
                        writer_4SaveAsPic=self.writer_4SaveAsPic
                    )
                    torch.save(lastest_state, unet_lastest_path)

                # 最优模型保存，用于测试
                if epoch >= (self.num_epochs//2) and unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_state = dict(
                        model=self.unet.state_dict()
                    )
                    self.myprint('Best model in epoch %d, score : %.4f' % (best_epoch + 1, best_unet_score))
                    torch.save(best_state, unet_best_path)

                # save_record_in_xlsx
                if (True):
                    excel_save_path = os.path.join(self.result_path, 'record.xlsx')
                    # record = pd.ExcelWriter(excel_save_path)
                    # detail_result1 = pd.DataFrame(valid_record)
                    # detail_result1.to_excel(record, 'valid', float_format='%.5f')
                    # record.save()
                    # record.close()
                    detail_result1 = pd.DataFrame(valid_record)
                    with pd.ExcelWriter(excel_save_path, engine='openpyxl') as record:
                        detail_result1.to_excel(record, 'valid', float_format='%.5f')

            # 规律性保存，可保存最终模型，用于测试
            if (epoch + 1) % self.save_model_step == 0:
                save_state = dict(
                    model=self.unet.state_dict()
                )
                torch.save(save_state, os.path.join(self.model_path, 'epoch%d.pkl' % (epoch + 1)))

            # 清理无用的内存
            del images, GT, SR, SR_probs, SR_flat, GT_flat, loss, loss_soft_dice, loss_bi_BCE
            torch.cuda.empty_cache()
            gc.collect()
            #
            toc = datetime.datetime.now()
            h, remainder = divmod((toc - tic).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "per epoch testing&vlidation cost Time %02d h:%02d m:%02d s" % (h, m, s)
            print(char_color(time_str))

        self.myprint('Finished!')
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        self.myprint('Best model in epoch %d, score : %.4f' % (best_epoch + 1, best_unet_score))

    def test(self, mode='train', unet_path=None):
        """Test model & Calculate performances."""
        if not unet_path is None:
            if os.path.isfile(unet_path):
                best_status = torch.load(unet_path)
                self.unet.load_state_dict(best_status['model'], False)
                self.myprint('Best model is Successfully Loaded from %s' % unet_path)

        self.unet.train(False)
        self.unet.eval()

        if mode == 'train':
            data_lodear = self.train_loader
            batch_size_test = self.batch_size
        elif mode == 'valid' or mode == 'val':
            data_lodear = self.valid_loader
            batch_size_test = self.batch_size_test

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        DC = 0.  # Dice Coefficient
        IOU = 0.  # IOU
        length = 0

        # model pre for each image
        detail_result = []  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
        with torch.no_grad():
            for i, sample in enumerate(data_lodear):
                (image_paths, images, GT) = sample
                images_path = list(image_paths)
                images = images.to(self.device)
                GT = GT.to(self.device)

                # modelsize(self.unet, images)
                if not self.TTA:
                    # no TTA
                    SR = self.unet(images[:, 0:1, :, :])
                    SR = F.sigmoid(SR)
                else:
                    # TTA
                    transforms = tta.Compose(
                        [
                            tta.VerticalFlip(),
                            tta.HorizontalFlip(),
                            tta.Rotate90(angles=[0, 180])
                            # tta.Scale(scales=[1, 2])
                            # tta.Multiply(factors=[0.9, 1, 1.1]),a
                        ]
                    )
                    tta_model = tta.SegmentationTTAWrapper(self.unet, transforms)
                    SR_mean = tta_model(images)
                    SR = F.sigmoid(SR_mean).float()

                if self.save_image:
                    
                    images_all = torch.cat((images[:, 0:1, :, :], SR[:, 0:1, :, :], SR[:, 1:2, :, :],
                                            GT[:, 0:1, :, :],GT[:, 1:2, :, :]), 0)
                    torchvision.utils.save_image(images_all.data.cpu(), os.path.join(self.result_path, 'images',
                                                                                     '%s_%d_image.png' % (mode, i)),
                                                 nrow=batch_size_test)

                SR = SR.data.cpu().numpy()
                GT = GT.data.cpu().numpy()

                for ii in range(SR.shape[0]):
                    SR_tmp = SR[ii, :].reshape(-1)
                    GT_tmp = GT[ii, :].reshape(-1)

                    tmp_index = images_path[ii].split('/')[-1]
                    # tmp_index = images_path[ii].split('\\')[-1]  #windows
                    tmp_index = tmp_index.split('.')[0][:]
                    tmp_index = int(tmp_index)


                    SR_tmp = torch.from_numpy(SR_tmp).to(self.device)
                    GT_tmp = torch.from_numpy(GT_tmp).to(self.device)

                    # acc, se, sp, pc, dc, _, _, iou = get_result_gpu(SR_tmp, GT_tmp) 	# 少楠写的
                    result_tmp1 = get_result_gpu(SR_tmp, GT_tmp)

                    result_tmp = np.array([tmp_index,
                                           result_tmp1[0],
                                           result_tmp1[1],
                                           result_tmp1[2],
                                           result_tmp1[3],
                                           result_tmp1[4],
                                           result_tmp1[7]])
                    # print(result_tmp)
                    acc += result_tmp[1]
                    SE += result_tmp[2]
                    SP += result_tmp[3]
                    PC += result_tmp[4]
                    DC += result_tmp[5]
                    IOU += result_tmp[6]
                    detail_result.append(result_tmp)

                    length += 1
            # 手动清理不再需要的变量
            del images, GT, SR, SR_tmp, GT_tmp, result_tmp1, result_tmp
            torch.cuda.empty_cache()  # 释放GPU缓存
            gc.collect()  # 垃圾回收



        accuracy = acc / length
        sensitivity = SE / length
        specificity = SP / length
        precision = PC / length
        disc = DC / length
        iou = IOU / length
        detail_result = np.array(detail_result)

        if (self.save_detail_result):  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
            if mode == 'train':
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result.xlsx')
            elif mode == 'test' and self.TTA:
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result_TTA.xlsx')
            else:
                excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result_test.xlsx')
            # writer = pd.ExcelWriter(excel_save_path)
            # detail_result = pd.DataFrame(detail_result)
            # detail_result.to_excel(writer, mode, float_format='%.5f')
            # writer.save()
            # writer.close()
            with pd.ExcelWriter(excel_save_path, engine='openpyxl') as result_writer:
                detail_result_df = pd.DataFrame(detail_result)
                detail_result_df.to_excel(result_writer, mode, float_format='%.5f')

        return accuracy, sensitivity, specificity, precision, disc, iou
