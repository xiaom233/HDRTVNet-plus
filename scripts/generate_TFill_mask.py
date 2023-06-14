import os
import cv2
import numpy as np
import os.path as osp

r=0.75

in_path = r'/data2/xychen/Youtube_hdr/train_hdr_sub_256'
out_path = r'/data2/xychen/Youtube_hdr/train_hdr_sub_256'
#
# in_path = r'E:\dataset\Youtube_hdr\test_LE_pred'
# out_path = r'E:\dataset\Youtube_hdr\test_LE_pred_gray'

if not osp.exists(out_path):
    os.mkdir(out_path)

for filename in sorted(os.listdir(in_path)):
    print(filename)
    img_LQ = cv2.imread(osp.join(in_path, filename), -1)
    print(img_LQ.max())
    img_LQ = img_LQ.astype(np.float32) / 65535.
    # print(img_LQ)
    # H, W, C = img_LQ.shape
    # if H%32!=0 or W%32!=0:
    #     H_new = int(np.ceil(H / 32) * 32)
    #     W_new = int(np.ceil(W / 32) * 32)
    #     img_LQ = cv2.resize(img_LQ, (W_new, H_new))

    mask = np.max(img_LQ, 2)
    mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))
    mask = np.where(mask > 0.1, 1, 0)
    # mask = np.expand_dims(mask, 2).repeat(C, axis=2)
    # cv2.imshow('mask', mask)
    mask *= 255

    img_gray = mask.astype(np.uint8)
    print(os.path.join(out_path,filename).replace('.png', '_mask000.png'))
    cv2.imwrite(os.path.join(out_path,filename).replace('.png', '_mask000.png'), mask)
