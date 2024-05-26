import torch.nn.functional as F
import torch
import numpy as np


def mae(input, target):
    return torch.mean(torch.abs(input - target))


def logmae_wav(model, output_dict, target):
    loss = torch.log10(torch.clamp(mae(output_dict['wav'], target), 1e-8, np.inf))
    return loss


def get_loss_func(loss_type):
    if loss_type == 'logmae_wav':
        return logmae_wav

    elif loss_type == 'mae':
    	return mae

    else:
        raise Exception('Incorrect loss_type!')

def sisnr_loss(x, s, eps=1e-8, zero_mean: bool = True):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor, estimate value
          s: reference signal, N x S tensor, True value
    Return:
          sisnr: N tensor
    """
    if x.shape != s.shape:
        if x.shape[-1] > s.shape[-1]:
            x = x[:, :s.shape[-1]]
        else:
            s = s[:, :x.shape[-1]]
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    # x = torch.from_numpy(x)
    # x = torch.from_numpy(s)
    if zero_mean:
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    else:
        x_zm = x
        s_zm = s

    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    ans = 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    ans_ = - ans.mean()
    return ans_
