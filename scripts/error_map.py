import os
import cv2
import numpy as np
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt

r=0.75

in_path = r'C:\Users\Hoven_Li\Desktop\HDRTVNet\teaser\GT'
GT_path = r'C:\Users\Hoven_Li\Desktop\HDRTVNet\teaser\GT'
out_path = r'C:\Users\Hoven_Li\Desktop\HDRTVNet\teaser\Full'

if not osp.exists(out_path):
    os.mkdir(out_path)

for filename in sorted(os.listdir(in_path)):
    print(filename)
    img_LQ = cv2.imread(osp.join(in_path, filename), cv2.IMREAD_UNCHANGED)
    img_GT = cv2.imread(osp.join(GT_path, filename), cv2.IMREAD_UNCHANGED)

    img_LQ = img_LQ.astype(np.float32) / 65535.
    img_GT = img_GT.astype(np.float32) / 65535.
    h,w,c = img_GT.shape
    print(img_GT.shape)
    img_error = img_GT - img_LQ
    img_error = np.abs(img_error)
    img_error = np.mean(img_error, axis=2)
    # img_error = img_error * 255 * 10.
    # img_error = img_error.astype(np.uint8)
    print(img_error.shape)
    plt.figure()
    sns.set()
    ax = sns.heatmap(img_error, square=True, cbar=False, robust=True, xticklabels=False, yticklabels=False, cmap='YlGnBu')
    print(ax)
    plt.savefig(os.path.join(out_path, filename), bbox_inches='tight', pad_inches=0.0)
    # cv2.imwrite(os.path.join(out_path, filename), img_error)
