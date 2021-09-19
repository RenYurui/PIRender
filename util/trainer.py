import random
import importlib
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from util.distributed import master_only_print as print
from util.init_weight import weights_init

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_trainer(opt, net_G, net_G_ema, opt_G, sch_G, train_dataset):
    module, trainer_name = opt.trainer.type.split('::')

    trainer_lib = importlib.import_module(module)
    trainer_class = getattr(trainer_lib, trainer_name)
    trainer = trainer_class(opt, net_G, net_G_ema, opt_G, sch_G, train_dataset)
    return trainer

def get_model_optimizer_and_scheduler(opt):
    gen_module, gen_network_name = opt.gen.type.split('::')
    lib = importlib.import_module(gen_module)
    network = getattr(lib, gen_network_name)
    net_G = network(**opt.gen.param).to(opt.device)
    init_bias = getattr(opt.trainer.init, 'bias', None)
    net_G.apply(weights_init(
        opt.trainer.init.type, opt.trainer.init.gain, init_bias))

    net_G_ema = network(**opt.gen.param).to(opt.device)
    net_G_ema.eval()
    accumulate(net_G_ema, net_G, 0)
    print('net [{}] parameter count: {:,}'.format(
        'net_G', _calculate_model_size(net_G)))
    print('Initialize net_G weights using '
          'type: {} gain: {}'.format(opt.trainer.init.type,
                                     opt.trainer.init.gain))


    opt_G = get_optimizer(opt.gen_optimizer, net_G)

    if opt.distributed:
        net_G = nn.parallel.DistributedDataParallel(
            net_G,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # Scheduler
    sch_G = get_scheduler(opt.gen_optimizer, opt_G)
    return net_G, net_G_ema, opt_G, sch_G


def _calculate_model_size(model):
    r"""Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_scheduler(opt_opt, opt):
    """Return the scheduler object.

    Args:
        opt_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if opt_opt.lr_policy.type == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=opt_opt.lr_policy.step_size,
            gamma=opt_opt.lr_policy.gamma)
    elif opt_opt.lr_policy.type == 'constant':
        scheduler = lr_scheduler.LambdaLR(opt, lambda x: 1)
    else:
        return NotImplementedError('Learning rate policy {} not implemented.'.
                                   format(opt_opt.lr_policy.type))
    return scheduler


def get_optimizer(opt_opt, net):
    return get_optimizer_for_params(opt_opt, net.parameters())


def get_optimizer_for_params(opt_opt, params):
    r"""Return the scheduler object.

    Args:
        opt_opt (obj): Config for the specific optimization module (gen/dis).
        params (obj): Parameters to be trained by the parameters.

    Returns:
        (obj): Optimizer
    """
    # We will use fuse optimizers by default.
    if opt_opt.type == 'adam':
        opt = Adam(params,
                   lr=opt_opt.lr,
                   betas=(opt_opt.adam_beta1, opt_opt.adam_beta2))
    else:
        raise NotImplementedError(
            'Optimizer {} is not yet implemented.'.format(opt_opt.type))
    return opt


