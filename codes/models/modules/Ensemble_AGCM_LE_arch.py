import torch
import torch.nn as nn
from .Condition_arch import ConditionNet
from .HDRUNet3T1_arch import HDRUNet3T1


class Ensemble_AGCM_LE(nn.Module):
    def __init__(self, classifier='color_condition', cond_c=6, in_nc=3, out_nc=3, nf=32, act_type='relu', weighting_network=False):
        super(Ensemble_AGCM_LE, self).__init__()
        self.AGCM = ConditionNet(classifier=classifier, cond_c=cond_c)
        # fix AGCM
        # for p in self.parameters():
        #     p.requires_grad = False

        self.LE = HDRUNet3T1(in_nc=in_nc, out_nc=out_nc, nf=nf, act_type=act_type, weighting_network=weighting_network)

    def forward(self, x):
        condition_output, input = self.AGCM(x)
        LE_input = [condition_output, condition_output]
        # condition = image
        LE_output = self.LE(LE_input)
        return LE_output[0], condition_output

if __name__=='__main__':
    net = Ensemble_AGCM_LE()
    net = net.state_dict()
    crt_net_keys = set(net.keys())
    print(crt_net_keys)