import numpy as np
import cv2

mask_path = 'C:\\Users\\Hoven_Li\\Desktop\\HDRTVNet\\mask\\001.npy'
GT_path = 'C:\\Users\\Hoven_Li\\Desktop\\HDRTVNet\\mask\\001.npy\\001_hdr.png'
LQ_path = 'C:\\Users\\Hoven_Li\\Desktop\\HDRTVNet\\mask\\001.npy\\001_sdr.png'
txt = 'C:\\Users\\Hoven_Li\\Desktop\\HDRTVNet\\mask\\mask.txt'

cond = np.load(mask_path, allow_pickle=True).astype(np.float32)
np.savetxt(txt, cond)
cv2.imshow('mask', cond)
cv2.waitKey()