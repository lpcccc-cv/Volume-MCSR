import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss, LapLoss, ncc_loss, fftLoss, MutualInformation, Mutual_info_reg, GradLoss
from matplotlib import pyplot as plt
import os
import numpy as np

logger = logging.getLogger('base')


def vis_img(img, fname, ftype ,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(img, cmap='gray')
    figname = fname + '_' + ftype + '.png'
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)

class BaseModel(BaseModel):
    def __init__(self, opt):
        super(BaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.which_dataset = opt["mode"]
        self.scale = opt["scale"]
        self.task = opt["task"]

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='mean').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)

                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            self.l1_loss = nn.L1Loss(reduction='mean').to(self.device)
            self.mi_loss = MutualInformation().to(self.device)


            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)
            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    
    def select_generate_slices(self, volume, scale):
        indices_to_remove = [i*scale for i in range(25)]
        indices_to_keep = [idx for idx in range(volume.shape[1]) if idx not in indices_to_remove]
        slice_generated = torch.index_select(volume, dim=1, index=torch.tensor(indices_to_keep).to(volume.device))
        return slice_generated
    
    def feed_data(self, data):   
        self.volume_LQ = data['volume_LQ'].to(self.device)
        self.volume_GT = data['volume_GT'].to(self.device)
        self.volume_ref = data['volume_ref'].to(self.device)
        
        if self.task == 'sr':
            self.volume_GT = self.volume_GT[:,1:-1]
        elif self.task == 'sr_volume':
            self.volume_GT = self.select_generate_slices(self.volume_GT, self.scale)


    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

      
    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        
        self.rec_ref = None
        self.fake_1, self.rec_ref = self.netG(self.volume_LQ, self.volume_ref)
        # self.fake_1 = self.netG(self.volume_LQ, self.volume_ref)

        # L1 loss
        l_pix = self.cri_pix(self.fake_1, self.volume_GT)

        if self.rec_ref != None:
            l_ref = self.cri_pix(self.rec_ref, self.volume_GT) 
            total_loss = l_pix + 0.1*l_ref
        else:
            total_loss = l_pix
        
        total_loss.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, _ = self.netG(self.volume_LQ, self.volume_ref)
            # self.fake_H = self.netG(self.volume_LQ, self.volume_ref)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    
    def get_current_visuals(self, need_GT=True):

        out_dict = OrderedDict()
        out_dict['im1_restore'] = self.fake_H.detach().float().cpu()
        out_dict['im1_GT'] = self.volume_GT.detach().float().cpu() 
        
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label, epoch=None):
        if epoch != None:
            self.save_network(self.netG, 'G', iter_label, str(epoch))
        else:
            self.save_network(self.netG, 'G', iter_label)
