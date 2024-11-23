from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.knn.__init__ import KNearestNeighbor


def loss_calculation(pred_r, pred_t, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
    knn = KNearestNeighbor(1)
    bs, num_p, _ = pred_r.size()

    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)

    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    points = points.contiguous().view(bs * num_p, 1, 3)

    pred = torch.add(torch.bmm(model_points, base), points + pred_t)

    if not refine:
        if idx[0].item() in sym_list:
            target = target[0].transpose(1, 0).contiguous().view(3, -1)
            pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
            inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
            target = torch.index_select(target, 1, inds.view(-1) - 1)
            target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.mean(torch.mean(torch.norm((pred - target), dim=2), dim=1), dim=0)

    del knn
    return dis


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points, w, refine):
        return loss_calculation(pred_r, pred_t, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)