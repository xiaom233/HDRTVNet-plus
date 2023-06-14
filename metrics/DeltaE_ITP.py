import colour
import cv2
import numpy as np
import math
import os
import torch
import torch.nn.functional as F
from piq import srsim
from piq import ssim



def calculate_hdr_deltaITP_released(img1, img2):
    img1 = img1[:, :, [2, 1, 0]]
    img2 = img2[:, :, [2, 1, 0]]
    img1 = img1 / 65535.
    img2 = img2 / 65535.
    img1 = colour.models.eotf_ST2084(img1)
    img2 = colour.models.eotf_ST2084(img2)
    img1_ictcp = colour.RGB_to_ICTCP(img1)
    img2_ictcp = colour.RGB_to_ICTCP(img2)
    delta_ITP = 720 * np.sqrt((img1_ictcp[:, :, 0] - img2_ictcp[:, :, 0]) ** 2
                              + 0.25 * ((img1_ictcp[:, :, 1] - img2_ictcp[:, :, 1]) ** 2)
                              + (img1_ictcp[:, :, 2] - img2_ictcp[:, :, 2]) ** 2)
    return np.mean(delta_ITP)


img_GT_path = r'/data/zyli/datasets/Youtube_hdr/test_hdr/'
img_path_root = r'/data/zyli/projects/HDRTVNet_New/results/060_Ensemble_AGCM_LE_490k/test_set/'
deltaITP_list = []
SRSIM_list = []
SSIM_list = []
print(img_path_root)
img_path = sorted(os.listdir(img_GT_path))
for i in range(len(img_path)):
    img_name = img_path[i]

    img_GT = cv2.imread(os.path.join(img_GT_path, img_name), cv2.IMREAD_UNCHANGED)
    img = cv2.imread(os.path.join(img_path_root, img_name), cv2.IMREAD_UNCHANGED)
    img_GT = img_GT.astype(np.int32)
    img = img.astype(np.int32)

    tensor_GT = torch.Tensor(img_GT).unsqueeze(0).permute(0, 3, 1, 2)
    tensor_pred = torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2)

    deltaITP = calculate_hdr_deltaITP_released(img, img_GT)
    print('{}, deltaITP: {:.6f}'.format(img_name, deltaITP))
    deltaITP_list.append(deltaITP)

    SRSIM = srsim(tensor_GT, tensor_pred, reduction='mean', data_range=65535.)
    print('{}, SR-SIM: {:.6f}'.format(img_name, SRSIM))
    SRSIM_list.append(SRSIM)

    SSIM = ssim(tensor_GT, tensor_pred, reduction='mean', data_range=65535., downsample=False)
    print('{}, SSIM: {:.6f}'.format(img_name, SSIM))
    SSIM_list.append(SSIM)

print(img_path_root)
print('deltaITP: {:f}'.format(sum(deltaITP_list) / len(deltaITP_list)))
print('SR-SIM: {:f}'.format(sum(SRSIM_list) / len(SRSIM_list)))
print('SSIM: {:f}'.format(sum(SSIM_list) / len(SSIM_list)))


