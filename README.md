# HDRTVNet [[Paper Link]](https://arxiv.org/abs/2108.07978)

### A New Journey from SDRTV to HDRTV
Xiangyu Chen*, Zhengwen Zhang*, [Jimmy S. Ren](https://scholar.google.com.hk/citations?hl=zh-CN&user=WKO_1VYAAAAJ), Lynhoo Tian, [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=zh-CN) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

(* indicates equal contribution)

**This paper is accepted to ICCV 2021.**

**I will give a detailed interpretation about this work on Zhihu. The link will be released soon.**

## Overview
Simplified SDRTV/HDRTV formation pipeline:

<img src="https://raw.githubusercontent.com/chxy95/HDRTVNet/master/figures/Formation_Pipeline.png" width="600"/>

Overview of the method:

<img src="https://raw.githubusercontent.com/chxy95/HDRTVNet/master/figures/Network_Structure.png" width="900"/>

## Getting Started

1. [Dataset](#dataset)
2. [Configuration](#configuration)
3. [How to test](#how-to-test)
4. [How to train](#how-to-train)
5. [Metrics](#metrics)
6. [Visualization](#visualization)

### Dataset
We conduct a dataset using videos with 4K resolutions under HDR10 standard (10-bit, Rec.2020, PQ) and their counterpart SDR versions from Youtube. The dataset consists of a training set with 1235 image pairs and a test set with 117 image pairs. Please refer to the paper for the details on the processing of the dataset. The dataset can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1TwXnBzeV6TlD3zPvuEo8IQ) (access code: 6qvu) or [OneDrive](https://uofmacau-my.sharepoint.com/:f:/g/personal/yc17494_umac_mo/Ep6XPVP9XX9HrZDUR9SmjdkB-t1NSAddMIoX3iJmGwqW-Q?e=dNODeW) (access code: HDRTVNet). The training set is uploaded after subsection compression since it's too large. Please download the complete dataset to unzip.

We also provide the original Youtube links of these videos, which can be found in this [**file**](https://raw.githubusercontent.com/chxy95/HDRTVNet/master/video_links.txt). Note that we cannot provide the download links since we do not have the copyright to distribute. **Please download this dataset only for academic use.**

### Configuration

Please refer to the [requirements](https://raw.githubusercontent.com/chxy95/HDRTVNet/master/requirements.txt). Matlab is also used to process the data, but it is not necessary and can be replaced by OpenCV.

### How to test

We provide the pretrained models to test, which can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1OSLVoBioyen-zjvLmhbe2g) (access code: 2me9) or [OneDrive](https://uofmacau-my.sharepoint.com/:f:/g/personal/yc17494_umac_mo/EteMb8FVYE5GqILE2mV-1W8B0-S_ynjt2gAgHkDH9LgkMg?e=EnBn3Q) (access code: HDRTVNet). Since our method is casaded of three steps, the results also need to be inferenced step by step. 
- Before testing, it is optional to generate the downsampled inputs of the condition network in advance. Make sure the `input_folder` and `save_LR_folder` in `./scripts/generate_mod_LR_bic.m` are correct, then run the file using Matlab. After that, matlab-bicubic-downsampled versions of the input SDR images are generated that will be input to the condition network. Note that this step is not necessary, but can reproduce more precise performance. Besides, if the pretrained HG model would be used, you should generate the masks of the input SDR images using `./scripts/generate_mask.py` and modify the corresponding paths in the config files.
- For the first part of AGCM, make sure the paths of `dataroot_LQ`, `dataroot_cond`, `dataroot_GT` and `pretrain_model_G` in `./codes/options/test/test_AGCM.yml` are correct, then run
```
cd codes
python test.py -opt options/test/test_AGCM.yml
```
- Note that if the first step is not preformed, the line of `dataroot_cond` should be commented. The test results will be saved to `./results/Adaptive_Global_Color_Mapping`.
- For the second part of LE, make sure `dataroot_LQ` is modified into the path of results obtained by AGCM, then run 
```
python test.py -opt options/test/test_LE.yml
```
- Note that results generated by LE can achieve the best quantitative performance. The part of HG is for the completeness of the solution and improving the visual quality forthermore. For testing the last part of HG, make sure `dataroot_LQ` is modified into the path of results obtained by LE, then run 
```
python test.py -opt options/test/test_HG.yml
```
- Note that the results of the each step are 16-bit images that can be converted into HDR10 video.

### How to train

- Prepare the data. Generate the sub-images with specific patch size using `./scripts/extract_subimgs_single.py` and generate the down-sampled inputs for the condition network (using the `./scripts/generate_mod_LR_bic.m` or any other methods). As the pipeline of the testing process, masks of SDR images need to be generated in advance if the HG part would be trained.
- For AGCM, make sure that the paths and settings in `./options/train/train_AGCM.yml` are correct, then run
```
cd codes
python train.py -opt options/train/train_AGCM.yml
```
- For LE, the inputs are generated by the trained AGCM model. The original data should be inferenced through the first step (refer to the last part on how to test AGCM) and then be processed by extracting sub-images. After that, modify the corresponding settings in `./options/train/train_LE.yml` and run
```
python train.py -opt options/train/train_LE.yml
```
- For HG, the inputs are also obtained by the last part LE, thus the training data need to be processed by similar operations as the previous two parts. When the data is prepared, it is recommended to pretrain the generator at first by running
```
python train.py -opt options/train/train_HG_Generator.yml
```
- After that, choose a pretrained model and modify the path of pretrained model in `./options/train/train_HG_GAN.yml`, then run
```
python train.py -opt options/train/train_HG_GAN.yml
```
- All models and training states are stored in `./experiments`.

### Metrics

Five metrics are used to evaluate the quantitative performance of different methods, including PSNR, SSIM, SR_SIM, Delta E<sub>ITP</sub> [(ITU Rec.2124)](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2124-0-201901-I!!PDF-E.pdf) and [HDR-VDP3](https://sourceforge.net/projects/hdrvdp/). Since the latter three metrics are not very common in recent papers, we provide some reference codes in `./metrics` for convenient usage.

### Visualization

Since HDR10 is an HDR standard using PQ transfer function for the video, the correct way to visualize the results is to synthesize the image results into a video format and display it on the HDR monitor or TVs that support HDR. The HDR images in our dataset are generated by directly extracting frames from the original HDR10 videos, thus these images consisting of PQ values look relatively dark compared to their true appearances. We provide the reference commands of our [**extracting**](https://github.com/chxy95/HDRTVNet/blob/main/scripts/extract_frames.sh) frames and [**synthesizing**](https://github.com/chxy95/HDRTVNet/blob/main/scripts/synthesizing_hdr10_video.sh) videos in `./scripts`. Please use [MediaInfo](https://mediaarea.net/en/MediaInfo) to check the format and the encoding information of synthesized videos before visualization. If circumstances permit, we strongly recommend to observe the HDR results and the original HDR resources by this way on the HDR dispalyer. 

If the HDR displayer is not available, some media players with HDR render can play the HDR video and show a relatively realistic look, such as [Potplayer](https://potplayer.daum.net/). Note that this is only an approximate alternative, and it still cannot fully restore the appearance of HDR content on HDR monitors.

## Citation
If our work is helpful to you, please cite our paper:

    @InProceedings{chen2021hdrtvnet,
        author    = {Chen, Xiangyu and Zhang, Zhengwen and Ren, Jimmy S. and Tian, Lynhoo and Qiao, Yu and Dong, Chao},
        title     = {A New Journey From SDRTV to HDRTV},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2021},
        pages     = {4500-4509}
    }

## Acknowledgment
The code is inspired by [BasicSR](https://github.com/xinntao/BasicSR).