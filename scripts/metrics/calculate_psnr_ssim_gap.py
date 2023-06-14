import argparse
import cv2
import numpy as np
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all_restored1 = []
    ssim_all_restored1 = []
    psnr_all_restored2 = []
    ssim_all_restored2 = []

    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_restored1 = sorted(list(scandir(args.restored1, recursive=True, full_path=True)))
    img_list_restored2 = sorted(list(scandir(args.restored2, recursive=True, full_path=True)))

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    # restored1 PSNR and SSIM
    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        print(basename)
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if args.suffix == '':
            img_path_restored = img_list_restored1[i]
        else:
            img_path_restored = osp.join(args.restored1, basename + args.suffix + ext)
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')

        psnr_all_restored1.append(psnr)
        ssim_all_restored1.append(ssim)

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        print(basename)
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if args.suffix == '':
            img_path_restored2 = img_list_restored2[i]
        else:
            img_path_restored2 = osp.join(args.restored2, basename + args.suffix + ext)
        img_restored = cv2.imread(img_path_restored2, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')

        psnr_all_restored2.append(psnr)
        ssim_all_restored2.append(ssim)

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        pnsr_gap = psnr_all_restored1[i] - psnr_all_restored2[i]
        ssim_gap = ssim_all_restored1[i] - ssim_all_restored2[i]
        print(f'{i + 1:3d}: {basename:25}. \tPSNR: {pnsr_gap:.6f} dB, \tSSIM: {ssim_gap:.6f}')

    print(args.gt)
    print(args.restored1)
    print(args.restored2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='/data/zyli/datasets/Youtube_hdr/test_hdr/', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored1', type=str, default='/data/zyli/projects/HDRTVNet_New/results/040_HDRUNet3T1_nf32_woWN_withCond_GT240B8_COS_1e-4_670k/test_set/', help='Path to restored images')
    parser.add_argument('--restored2', type=str, default='/data/zyli/projects/HDRTVNet_New/results/059_SwinUNet_UNet_GAN_padding_400k/test_set/', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)
