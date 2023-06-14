import os
import cv2
import numpy as np
import os.path as osp


in_path = r'D:\科研\数据集\HDRTVNet\test_set\test_sdr_mask'
out_path = r'C:\Users\Hoven_Li\Desktop\test_sdr_mask_png'

if not osp.exists(out_path):
    os.mkdir(out_path)

for filename in sorted(os.listdir(in_path)):
    img = cond = np.load(os.path.join(in_path,filename), allow_pickle=True).astype(np.float32)
    img *= 255
    img_gray = img.astype(np.uint8)
    print(os.path.join(out_path,filename).replace('.npy', '.png'))
    cv2.imwrite(os.path.join(out_path,filename).replace('.npy', '.png'), img)