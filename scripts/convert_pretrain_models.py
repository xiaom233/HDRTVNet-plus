from collections import OrderedDict
import torch
from copy import deepcopy
from models.modules.Ensemble_AGCM_LE_arch import Ensemble_AGCN_LE

AGCM_path = '/data/zyli/projects/HDRTVNet_New/experiments/012_L1_AGCM_NewMultistage_5m8m_4e-4/models/615000_G.pth'
LE_path = '/data/zyli/projects/HDRTVNet_New/experiments/040_HDRUNet3T1_nf32_woWN_withCond_GT240B8_COS_1e-4_archived_220910-150552/models/670000_G.pth'
save_path = '/data/zyli/projects/HDRTVNet_New/pretrained_models/Ensemble_AGCM_LE.pth'

AGCM_dic = torch.load(AGCM_path)
LE_dic = torch.load(LE_path)
save_dict = {}
output_net = OrderedDict()
# merge AGCM
for k, v in deepcopy(AGCM_dic).items():
    if k.startswith('module.'):
        output_net['AGCM.'+k[7:]] = v
    else:
        output_net['AGCM.'+k] = v

# merge LE
for k, v in deepcopy(LE_dic).items():
    if k.startswith('module.'):
        output_net['LE.'+k[7:]] = v
    else:
        output_net['LE.'+k] = v

net = Ensemble_AGCN_LE()
net.load_state_dict(output_net, strict=True)

# state_dict = net.state_dict()
save_dict= output_net
torch.save(save_dict, save_path)