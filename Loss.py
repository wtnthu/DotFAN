import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error)
        return loss

def classification_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    return F.cross_entropy(logit, target)
def binary_cross_loss(cls,label):
    return F.binary_cross_entropy_with_logits(cls, label, size_average=False) / cls.size(0)
def total_variation_loss(img):
    out = (
            torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
            torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    )
    return out
def l1_norm(input,label):
        return  torch.mean(torch.abs(input - label))
def gradient_p( real_x, fake_x, D):
    alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
    interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
    out,_ = D(interpolated)

    grad = torch.autograd.grad(outputs=out,
                               inputs=interpolated,
                               grad_outputs=torch.ones(out.size()).cuda(),
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    grad = grad.view(grad.size(0), -1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    return torch.mean((grad_l2norm - 1) ** 2)

def cal_identity_loss( fake_x, real_x,R):
    # gram_weights=0.01
    samples = [(fake_x, real_x)]
    g_loss_identity = 0.0
    for ff, rr in samples:

        fake_f = R(ff,mode='eval')
        real_f = R(rr,mode='eval')
        print(ff.shape)
        # fake_grams = self.grams(fake_f)
        # real_grams = self.grams(real_f)

        for kk in range(len(fake_f)):
            g_loss_identity += torch.mean(torch.abs(real_f[kk] - fake_f[kk]))
            # g_loss_identity += torch.mean(torch.abs(real_grams[kk] - fake_grams[kk])) * gram_weights

    return g_loss_identity