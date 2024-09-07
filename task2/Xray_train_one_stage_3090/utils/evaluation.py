import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np

# SR : Segmentation Result
# GT : Ground Truth


def confusion(SR, GT):
    SR = SR.view(-1)
    GT = GT.view(-1)
    confusion_vector = SR / GT

    TP = torch.sum(confusion_vector == 1).item()
    FP = torch.sum(confusion_vector == float('inf')).item()
    TN = torch.sum(torch.isnan(confusion_vector)).item()
    FN = torch.sum(confusion_vector == 0).item()

    return TP, FP, TN, FN


def get_result(SR, GT, threshold=0.5):  # 没用到gpu版本
    SR[SR > threshold] = 1
    SR[SR < 1] = 0
    confusion = confusion_matrix(GT, SR)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    acc = (TP + TN) / (float(TP + TN + FP + FN) + 1e-6)
    sensitivity = TP / (float(TP + FN) + 1e-6)
    Specificity = TN / (float(TN + FP) + 1e-6)
    precision = TP / (float(TP + FP) + 1e-6)
    F1 = 2*sensitivity*precision / (sensitivity + precision + 1e-6)
    JS = sum((SR + GT) == 2) / (sum((SR + GT) >= 1) + 1e-6)
    DC = 2 * sum((SR + GT) == 2) / (sum(SR) + sum(GT) + 1e-6)
    IOU = TP / (float(TP+FP+FN) + 1e-6)

    # print('Accuracy:', acc)
    # print('Sensitivity:', sensitivity)
    # print('Specificity:', Specificity)
    # print('precision:', precision)
    # print('F1', F1)
    # print('JS', JS)
    # print('DC', DC)
    # print('IOU', IOU)

    return acc, sensitivity, Specificity, precision, F1, JS, DC, IOU


def get_result_gpu(SR, GT, threshold=0.5):  # gpu版本
    SR[SR > threshold] = 1
    SR[SR < 1] = 0
    TP, FP, TN, FN = confusion(SR, GT)

    acc = (TP + TN) / (float(TP+TN+FP+FN) + 1e-6)
    sensitivity = TP / (float(TP+FN) + 1e-6)
    specificity = TN / (float(TN+FP) + 1e-6)
    precision = TP / (float(TP+FP) + 1e-6)
    F1 = 2*sensitivity*precision / (sensitivity + precision + 1e-6)
    JS = TP / (float(FP+TP+FN) + 1e-6)
    DC = 2*TP / (float(FP+2*TP+FN) + 1e-6)
    IOU = TP / (float(TP+FP+FN) + 1e-6)

    return acc, sensitivity, specificity, precision, F1, JS, DC, IOU


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)

    # tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(SR.size(0))

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1)) == 2
    FN = ((SR==0)+(GT==1)) == 2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)

    return SP


def get_precision(SR,GT,threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC


def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR.view(-1)
    GT = GT.view(-1)
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)

    JS = float(Inter)/(float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC


def get_IOU(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2
    FN = ((SR == 0) + (GT == 1)) == 2

    IOU = float(torch.sum(TP))/(float(torch.sum(TP+FP+FN)) + 1e-6)

    return IOU


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)
        smooth = 1

        # probs = F.sigmoid(logits)
        # m1 = probs.view(num, -1)
        # m2 = targets.view(num, -1)
        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
    

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

class SoftDiceLoss2(torch.nn.Module):
    def __init__(self):
        super(SoftDiceLoss2, self).__init__()

    def forward(self, inputs, targets):


        # Intersection is equivalent to True Positive count
        intersection = (inputs * targets).sum()
        # Sum of all input values and sum of all true values (count of positives)
        union = inputs.sum() + targets.sum()

        dice_score = (2. * intersection) / (union + 1e-8)

        return 1 - dice_score

class SoftDiceLoss3(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss3, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
                
        dc = dc.mean()

        return -dc


