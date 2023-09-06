from collections import OrderedDict
import torch
from copy import deepcopy
from models.modules.Ensemble_AGCM_LE_arch import Ensemble_AGCN_LE

AGCM_path = 'AGCM.pth'
LE_path = 'LE.pth'
save_path = 'Ensemble_AGCM_LE.pth'

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