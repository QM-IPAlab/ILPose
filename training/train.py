

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dest
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


import sys
sys.path.append("..")
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.linemod.dataset_1 import PoseDataset_1 as PoseDataset_linemod_1
from datasets.linemod.dataset_2 import PoseDataset_2 as PoseDataset_linemod_2
from lib.network import PoseNet
from lib.loss import Loss
from lib.utils import setup_logger
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='linemod', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default='/Linemod/Linemod_preprocessed', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.09, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.18, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=300, help='max number of epochs to train')

parser.add_argument('--resume_posenet', type=str, default='/Linemod/keyframe_50/06_04_01_02/pose_model_300_0.010901178275451319.pth', help='load previous trained model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()




class EWC:
    def __init__(self, model, dataloader, fisher_multiplier, memory_pool, Criterion):
        self.model = model
        self.dataloader = dataloader
        self.fisher_multiplier = fisher_multiplier
        self.loss = Criterion
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.optimal_params = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher = {n: p.clone().detach().zero_() for n, p in self.params.items()}
        self.memory_pool = memory_pool

    def update_fisher(self):
        self.model.eval()

        # Update Fisher information using the keyframes from the memory pool
        for i, data in enumerate(self.memory_pool, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(target).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda()
            self.model.zero_grad()
            pred_r, pred_t, emb = self.model(img, points, choose)
            loss = self.loss(pred_r, pred_t, target, model_points, idx, points, opt.w, opt.refine_start)
            loss.backward()
            for n, p in self.model.named_parameters():
                self.fisher[n] += (p.grad ** 2) / len(self.memory_pool)

    def penalty(self):
        loss = 0
        for n, p in self.params.items():
            _loss = self.fisher[n] * (p - self.optimal_params[n]) ** 2
            # _loss = (p - self.optimal_params[n]) ** 2
            loss += _loss.sum() * self.fisher_multiplier
        return loss


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 500  # number of points on the input pointcloud
        opt.outf = ''  # folder to save trained models
        opt.log_dir = ''  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training

    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = ''
        opt.log_dir = ''
        opt.repeat_epoch = 1

    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects)
    estimator.cuda()


    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}'.format(opt.resume_posenet)))
        print("#########  YES, pre-trained model from {} is loaded   #########".format(opt.resume_posenet.split('/')[-2]))


    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
        test_dataset_1 = PoseDataset_linemod_1('train', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    testdataloader_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset_1), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf


    st_time = time.time()


    ewc = EWC(estimator, dataloader, 60, testdataloader_1, criterion)


    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0

        estimator.train()
        optimizer.zero_grad()

        for i, data in enumerate(dataloader, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(target).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda()
            pred_r, pred_t, emb = estimator(img, points, choose)

            loss = criterion(pred_r, pred_t, target, model_points, idx, points, opt.w, opt.refine_start)

            ewc_loss = ewc.penalty()

            loss += ewc_loss

            loss.backward()

            train_count += 1
            print("train_count: {0}, loss_all: {1}, loss_ADD: {2}, loss_ewc: {3}".format(train_count, loss, loss.item(), ewc_loss.item()))


            if train_count % opt.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            if train_count != 0 and train_count % 1000 == 0:
                torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        ewc.update_fisher()
        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        ##### test the trained model on training set #####
        logger_2 = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_benchivise_log.txt' % epoch))
        logger_2.info('trained: Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(target).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda()
            pred_r, pred_t, emb = estimator(img, points, choose)
            loss_test = criterion(pred_r, pred_t, target, model_points, idx, points, opt.w, opt.refine_start)
            test_dis += loss_test.item()
            test_count += 1

        test_dis = test_dis / test_count
        logger_2.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))


        if test_dis <= best_test:
            best_test = test_dis
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        print("test finish")



if __name__ == '__main__':
    main()
