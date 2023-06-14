import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import copy
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from basicsr.metrics import calculate_niqe

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
tb_logger = SummaryWriter(log_dir=os.path.join(opt['path']['root'], 'tb_logger', opt['name']))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)

opt_dir = []
for i in range(5000, 400000, 5000):
    temp_opt = copy.deepcopy(opt)
    previous_pth_path = temp_opt['path']['pretrain_model_G'].split('/models/')
    current_pth_path = previous_pth_path[0] + '/models/' + str(i) + '_G.pth'
    temp_opt['path']['pretrain_model_G'] = current_pth_path
    opt_dir.append(temp_opt)

for current_opt in opt_dir:
    model.load_network(current_opt['path']['pretrain_model_G'], model.netG)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        print('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(current_opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['niqe'] = []

        for data in test_loader:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            model.feed_data(data, need_GT=need_GT)
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)

            sr_img = util.tensor2img(visuals['SR'], np.uint16)  # uint16

            # save images
            # suffix = opt['suffix']
            # if suffix:
            #     save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            # else:
            #     save_img_path = osp.join(dataset_dir, img_name + '.png')
            # util.save_img(sr_img, save_img_path)

            # calculate PSNR
            gt_img = util.tensor2img(visuals['GT'], np.uint16) # uint8 / uint16
            gt_img = gt_img / (pow(2,16)-1)
            sr_img = sr_img / (pow(2,16)-1)
            niqe = calculate_niqe(sr_img ,0, input_order='HWC', convert_to='y')
            test_results['niqe'].append(niqe)
            print('{:20s} - NIQE: {:.6f} dB'.format(img_name, niqe))
        # Average PSNR results
        current_iteration_model = current_opt['path']['pretrain_model_G'].split('/models/')
        current_iteration = current_iteration_model[1]
        temp = current_iteration.split('_')
        itration = temp[0]
        ave_psnr = sum(test_results['niqe']) / len(test_results['niqe'])
        logger.info('<path:{}> niqe: {:.4e}'.format(
            current_iteration, ave_psnr))
        tb_logger.add_scalar('niqe', ave_psnr, int(itration))
