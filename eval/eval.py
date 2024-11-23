import argparse
import os
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import open3d as o3d
from PIL import Image
import scipy.misc
import scipy.io as scio
import cv2

import sys
sys.path.append("..")
from datasets.linemod.dataset_vis import PoseDataset as PoseDataset_linemod
from lib.network_no_mask_v2 import PoseNet_v2
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def search_fit(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]

def build_frame(min_x, max_x, min_y, max_y, min_z, max_z):
    bbox = []
    for i in np.arange(min_x, max_x, 1.0):
        bbox.append([i, min_y, min_z])
    for i in np.arange(min_x, max_x, 1.0):
        bbox.append([i, min_y, max_z])
    for i in np.arange(min_x, max_x, 1.0):
        bbox.append([i, max_y, min_z])
    for i in np.arange(min_x, max_x, 1.0):
        bbox.append([i, max_y, max_z])

    for i in np.arange(min_y, max_y, 1.0):
        bbox.append([min_x, i, min_z])
    for i in np.arange(min_y, max_y, 1.0):
        bbox.append([min_x, i, max_z])
    for i in np.arange(min_y, max_y, 1.0):
        bbox.append([max_x, i, min_z])
    for i in np.arange(min_y, max_y, 1.0):
        bbox.append([max_x, i, max_z])

    for i in np.arange(min_z, max_z, 1.0):
        bbox.append([min_x, min_y, i])
    for i in np.arange(min_z, max_z, 1.0):
        bbox.append([min_x, max_y, i])
    for i in np.arange(min_z, max_z, 1.0):
        bbox.append([max_x, min_y, i])
    for i in np.arange(min_z, max_z, 1.0):
        bbox.append([max_x, max_y, i])
    bbox = np.array(bbox)

    return bbox



def compute_rotation_degree(pred_rot, gt_rot):
    gt_rot = gt_rot[0].cpu().detach().numpy()



    pred_angle = np.arccos((np.trace(pred_rot) - 1) / 2.0)
    pred_axis = 1 / (2 * np.sin(pred_angle)) * np.array([
        pred_rot[2, 1] - pred_rot[1, 2],
        pred_rot[0, 2] - pred_rot[2, 0],
        pred_rot[1, 0] - pred_rot[0, 1]
    ])

    gt_angle = np.arccos((np.trace(gt_rot) - 1) / 2.0)
    gt_axis = 1 / (2 * np.sin(gt_angle)) * np.array([
        gt_rot[2, 1] - gt_rot[1, 2],
        gt_rot[0, 2] - gt_rot[2, 0],
        gt_rot[1, 0] - gt_rot[0, 1]
    ])

    dot_product = np.dot(pred_axis, gt_axis)
    angle_between = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return np.degrees(angle_between)

def translation_distance(pred_t, gt_t):
    gt_t = gt_t[0].cpu().detach().numpy()
    gt_t = gt_t / 1000.0
    distance = np.linalg.norm(pred_t - gt_t)
    distance = distance
    return distance

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'Linemod/Linemod_preprocessed', help='dataset root dir')
parser.add_argument('--model', type=str, default = '/linemod/01_new/feature_estimator_model_1_0.00010009019752033055.pth',  help='resume trained model')
opt = parser.parse_args()



num_objects = 1
objlist = [1]

num_points = 500
iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNearestNeighbor(1)




estimator = PoseNet_v2(num_points = num_points, num_obj = num_objects)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()


testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
# meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta_file = open('/linemod/dataset_config/models_info.yml', 'r')
meta = yaml.load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]

diff_degree_all = [0 for i in range(num_objects)]
diff_distance_all = [0 for i in range(num_objects)]


fw = open('/experiments/eval_result/linemod/eval_result_logs.txt', 'w')


for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx, img_location, gt_r, gt_t = data

    img_location = img_location[0]
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                    Variable(choose).cuda(), \
                                                    Variable(img).cuda(), \
                                                    Variable(target).cuda(), \
                                                    Variable(model_points).cuda(), \
                                                    Variable(idx).cuda()

    pred_r, pred_t = estimator(img, points, choose)

    pred_r = pred_r / pred_r.norm(dim=1, keepdim=True)




    my_r = pred_r.view(-1).cpu().data.numpy()
    my_r = quaternion_matrix(my_r)[:3, :3]
    my_t = pred_t[0].view(-1).cpu().data.numpy()
    model_points = model_points[0].cpu().detach().numpy()


    diff_degree = compute_rotation_degree(my_r, gt_r)
    if np.isnan(diff_degree):
        diff_degree_all[idx[0].item()] += 0
    else:
        diff_degree_all[idx[0].item()] += diff_degree


    diff_distance = translation_distance(my_t, gt_t)
    if np.isnan(diff_distance):
        diff_distance_all[idx[0].item()] += 0
    else:
        diff_distance_all[idx[0].item()] += diff_distance




    pred = np.dot(model_points, my_r.T) + my_t
    target = target[0].cpu().detach().numpy()



    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))


    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1

for i in range(num_objects):
    print('object {0}, mean rotation error {1}, mean translation error {2} '.format(objlist[i], float(diff_degree_all[i] / num_count[i]), float(diff_distance_all[i] / num_count[i])))
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()








