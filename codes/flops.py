from thop import profile
import torch
from models.modules.HDRUNet3_arch import HDRUNet3 as net
#from models.modules.Base_arch import SRResNet as net
#from basicsr.archs.rfdnfinalB5_arch import RFDNFINALB5 as net

#model = net(upscale=4,in_chans=3,img_size=48, window_size=16, img_range=1., depths=[6, 6, 6, 6, 6, 6, 6], embed_dim=60, 
#          num_heads=[1, 1, 1, 1, 1, 1, 1], mlp_ratio=1, upsampler='pixelshuffle', resi_connection='1conv')

model = net(nf=50)
input = torch.randn(1, 3, 63, 63)
_, _, H, W = input.shape
mod = 8
#input = input[:, :, : -(H % mod), : -(W % mod)]
#output = model(input)
#output = model(input)
#print(output.shape)
flops, params = profile(model, inputs=((input, input), ))

print("FLOPs[G] ")
print(flops/1e9)
print("Parameters [K]")
print(params/1e3)
