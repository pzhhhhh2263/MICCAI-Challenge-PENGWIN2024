from torch import nn
import torch


# 用于自动学习loss之间的权重
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    https://arxiv.org/abs/1705.07115
    解读：https://blog.csdn.net/cv_family_z/article/details/78749992
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num=2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, device, weights_list=None, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        if weights_list is not None:
            params = torch.tensor(weights_list, requires_grad=True).to(device)
        else:
            params = torch.ones(num, requires_grad=True).to(device)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

    def get_weights(self):
        weights = list(self.params.data.cpu().numpy())
        weights = [1 / (i ** 2) for i in weights]
        return weights
