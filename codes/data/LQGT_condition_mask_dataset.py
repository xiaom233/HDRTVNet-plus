import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp
from torch.nn import functional as F


class LQGT_dataset(data.Dataset):

    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None

        self.sizes_GT, self.paths_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.sizes_LQ, self.paths_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.sizes_mask, self.paths_mask = util.get_image_paths(self.data_type, opt['dataroot_mask'])
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

        self.mask_folder = opt['dataroot_mask']
        self.cond_folder = opt['dataroot_cond']

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_img(None, GT_path)

        # get LQ image
        LQ_path = self.paths_LQ[index]
        img_LQ = util.read_img(None, LQ_path)

        # get mask of SDR
        mask_path = self.paths_mask[index]
        mask = util.read_npy(mask_path)
        mask = np.expand_dims(mask, 2).repeat(3, axis=2)

        # # get condition
        cond_scale = self.opt['cond_scale']
        if self.cond_folder is not None:
            if '_' in osp.basename(LQ_path):
                cond_name = '_'.join(osp.basename(LQ_path).split('_')[:-1])+'_bicx'+str(cond_scale)+'.npy'
            else: cond_name = osp.basename(LQ_path).split('.')[0]+'_bicx'+str(cond_scale)+'.png'
            cond_path = osp.join(self.cond_folder, cond_name)
            cond_img = util.read_img(None, cond_path)
        else:
            cond_img = util.imresize_np(img_LQ, 1/cond_scale)

        if self.opt['phase'] == 'train':

            H, W, C = img_LQ.shape
            H_gt, W_gt, C = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))
            LQ_size = GT_size // scale

            # randomly crop
            if GT_size is not None:
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                # img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
                img_LQ = img_LQ[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
                mask = mask[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT, mask = util.augment([img_LQ, img_GT, mask], self.opt['use_flip'],
                                                self.opt['use_rot'])
        else:
            H, W, C = img_LQ.shape
            H_gt, W_gt, C = img_GT.shape

            if H % 8 != 0:
                img_LQ = img_LQ[: -(H % 8), :, :]
            if W % 8 != 0:
                img_LQ = img_LQ[:, : -(W % 8), :]

            if H_gt % 8 != 0:
                img_GT = img_GT[: -(H_gt % 8), :, :]
                mask = mask[: -(H_gt % 8), :, :]
            if W_gt % 8 != 0:
                img_GT = img_GT[:, : -(W_gt % 8), :]
                mask = mask[:, : -(W_gt % 8), :]
        # resize for alignment
        # H, W, C = img_LQ.shape
        # if H%32!=0 or W%32!=0:
        #     H_new = int(np.ceil(H / 32) * 32)
        #     W_new = int(np.ceil(W / 32) * 32)
        #     img_LQ = cv2.resize(img_LQ, (W_new, H_new))
        #     img_GT = cv2.resize(img_GT, (W_new, H_new))
        #     mask = cv2.resize(mask, (W_new, H_new))
        # use the input LQ to calculate the mask.
        # if self.mask_folder is None:
        #     r = 0.95
        #     mask = np.max(img_LQ, 2)
        #     mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))
        #     mask = np.expand_dims(mask, 2).repeat(3, axis=2)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            mask = mask[:, :, [2, 1, 0]]
            cond_img = cond_img[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        mask = torch.from_numpy(np.ascontiguousarray(np.transpose(mask, (2, 0, 1)))).float()
        cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond_img, (2, 0, 1)))).float()
        # if self.opt['padding64']:
        #     mod_pad_h = 0
        #     mod_pad_w = 0
        #     C, H, W = img_LQ.shape
        #     if H % 64 != 0:
        #         mod_pad_h = 64 - H % 64
        #     if W % 64 != 0:
        #         mod_pad_w = 64 - W % 64
        #     img_GT = img_GT.unsqueeze(0)
        #     img_LQ = img_LQ.unsqueeze(0)
        #     mask = mask.unsqueeze(0)
        #     img_GT = F.pad(img_GT, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        #     img_LQ = F.pad(img_LQ, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        #     mask = F.pad(mask, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        #     img_GT = img_GT.squeeze(0)
        #     img_LQ = img_LQ.squeeze(0)
        #     mask = mask.squeeze(0)

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'mask': mask, 'LQ_path': LQ_path, 'GT_path': GT_path, 'cond': cond}

    def __len__(self):
        return len(self.paths_GT)
