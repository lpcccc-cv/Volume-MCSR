import models.modules.MV_LKAN as MV_LKAN


####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    scale = opt['scale']

    if which_model == 'MV_LKAN':
        netG = MV_LKAN.MV_LKAN(scale=scale)

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
