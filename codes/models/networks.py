import torch
import models.modules.swinir_arch as swinir_arch
import models.modules.Base_arch as Base_arch
import models.modules.HDRUNet_arch as HDRUNet_arch
import models.modules.HDRUNet3_arch as HDRUNet3_arch
import models.modules.HDRUNet4_arch as HDRUNet4_arch
import models.modules.HDRUNet3T1_arch as HDRUNet3T1_arch
import models.modules.HDRUNet3T2_arch as HDRUNet3T2_arch
import models.modules.Condition_arch as Condition_arch
import models.modules.Hallucination_arch as Hallucination_arch
import models.modules.swinir_arch as swinir_arch
import models.modules.SwinUNet_arch as SwinUNet_arch
import models.modules.discriminator_vgg_arch as DNet_arch
import models.modules.Discriminator_UNet_arch as Discriminator_UNet_arch
import models.modules.Ensemble_AGCM_LE_arch as Ensemble_AGCM_LE_arch
import models.modules.Ensemble_AGCM_LE_mask_arch as Ensemble_AGCN_LE_withMask
import models.modules.Condition_GAN_arch as ConditionNetGAN

import logging

logger = logging.getLogger('base')

####################
# define network
####################

#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'ConditionNet':
        netG = Condition_arch.ConditionNet(classifier=opt_net['classifier'], cond_c=opt_net['cond_c'])
    elif which_model == 'ConditionNetGAN':
        netG = ConditionNetGAN.ConditionNetGAN(classifier=opt_net['classifier'], cond_c=opt_net['cond_c'])
    elif which_model == 'ConditionNet2Layer':
        netG = Condition_arch.ConditionNet2Layer(classifier=opt_net['classifier'], cond_c=opt_net['cond_c'])
    elif which_model == 'ConditionNet4Layer':
        netG = Condition_arch.ConditionNet4Layer(classifier=opt_net['classifier'], cond_c=opt_net['cond_c'])
    elif which_model == 'ConditionNet5Layer':
        netG = Condition_arch.ConditionNet5Layer(classifier=opt_net['classifier'], cond_c=opt_net['cond_c'])
    elif which_model == 'BaseModel':
        netG = Condition_arch.BaseModel(nf=opt_net['nf'])
    elif which_model == 'BaseModel2layer':
        netG = Condition_arch.BaseModel2layer(nf=opt_net['nf'])
    elif which_model == 'BaseModel4layer':
        netG = Condition_arch.BaseModel4layer(nf=opt_net['nf'])
    elif which_model == 'BaseModel5layer':
        netG = Condition_arch.BaseModel5layer(nf=opt_net['nf'])
    elif which_model == 'SRResNet':
        netG = Base_arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                  nb=opt_net['nb'], act_type=opt_net['act_type'])
    elif which_model == 'HDRUNet':
        netG = HDRUNet_arch.HDRUNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                    act_type=opt_net['act_type'], weighting_network=opt_net['weighting_network'])
    elif which_model == 'HDRUNet3':
        netG = HDRUNet3_arch.HDRUNet3(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                    act_type=opt_net['act_type'], weighting_network=opt_net['weighting_network'])
    elif which_model == 'HDRUNet4':
        netG = HDRUNet4_arch.HDRUNet4(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                    act_type=opt_net['act_type'], weighting_network=opt_net['weighting_network'])
    elif which_model == 'HDRUNet3T1':
        netG = HDRUNet3T1_arch.HDRUNet3T1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                    act_type=opt_net['act_type'], weighting_network=opt_net['weighting_network'])
    elif which_model == 'HDRUNet3T2':
        netG = HDRUNet3T2_arch.HDRUNet3T2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                    act_type=opt_net['act_type'], weighting_network=opt_net['weighting_network'])
    elif which_model == 'vapsr_beta':
        netG = Base_arch.vapsr_beta(num_in_ch=opt_net['num_in_ch'], num_out_ch=opt_net['num_out_ch'], upfactor=opt_net['upfactor'],
                                 num_feat=opt_net['num_feat'], num_group=opt_net['num_group'], num_block_per_group=opt_net['num_block_per_group'],
                                 d_atten=opt_net['d_atten'])
    elif which_model == 'Hallucination_Generator':
        netG = Hallucination_arch.Hallucination_Generator(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'])
    elif which_model == 'SwinIR':
        netG = swinir_arch.SwinIR(
                 upscale=opt_net['upscale'],
                 in_chans=opt_net['in_chans'],
                 img_size=opt_net['img_size'],
                 window_size=opt_net['window_size'],
                 depths=opt_net['depths'],
                 embed_dim=opt_net['embed_dim'],
                 num_heads=opt_net['num_heads'],
                 mlp_ratio=opt_net['mlp_ratio'],
                 upsampler=opt_net['upsampler'],)
    elif which_model == 'SwinUNet':
        netG = SwinUNet_arch.SwinUNet(
            upscale=opt_net['upscale'],
            in_chans=opt_net['in_chans'],
            img_size=opt_net['img_size'],
            window_size=opt_net['window_size'],
            depths=opt_net['depths'],
            embed_dim=opt_net['embed_dim'],
            num_heads=opt_net['num_heads'],
            mlp_ratio=opt_net['mlp_ratio'],
            upsampler=opt_net['upsampler'], )
    elif which_model == 'Ensemble_AGCN_LE':
        netG = Ensemble_AGCM_LE_arch.Ensemble_AGCN_LE(
            classifier=opt_net['classifier'],
            cond_c=opt_net['cond_c'],
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            nf=opt_net['nf'],
            act_type=opt_net['act_type'],
            weighting_network=opt_net['weighting_network'])
    elif which_model == 'Ensemble_AGCN_LE_withMask':
        netG = Ensemble_AGCN_LE_withMask.Ensemble_AGCN_LE_withMask(
            classifier=opt_net['classifier'],
            cond_c=opt_net['cond_c'],
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            nf=opt_net['nf'],
            act_type=opt_net['act_type'],
            weighting_network=opt_net['weighting_network'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = DNet_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'Discriminator_UNet':
        netD = Discriminator_UNet_arch.Discriminator_UNet(input_nc=opt_net['input_nc'], ndf=opt_net['ndf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = DNet_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
