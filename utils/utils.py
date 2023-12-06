import torch
import random
import numpy as np
import logging
import time
import os


def compute_kernel_batch(x):
    batch_size = x.size(0)
    num_aug = x.size(1)
    dim = x.size(2)
    n_samples = batch_size * num_aug

    y = x.clone()
    x = x.unsqueeze(1).unsqueeze(3) 
    y = y.unsqueeze(0).unsqueeze(2)
    tiled_x = x.expand(batch_size, batch_size, num_aug, num_aug, dim)
    tiled_y = y.expand(batch_size, batch_size, num_aug, num_aug, dim)

    L2_distance = (tiled_x - tiled_y).pow(2).sum(-1)
    bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples)

    return torch.exp(-L2_distance / bandwidth)


def compute_mmd_batch(x):
    batch_size = x.size(0)
    batch_kernel = compute_kernel_batch(x) 
    batch_kernel_mean = batch_kernel.reshape(batch_size, batch_size, -1).mean(2)  
    self_kernel = torch.diag(batch_kernel_mean)
    x_kernel = self_kernel.unsqueeze(1).expand(batch_size, batch_size)
    y_kernel = self_kernel.unsqueeze(0).expand(batch_size, batch_size)
    mmd = x_kernel + y_kernel - 2*batch_kernel_mean

    return mmd.detach()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def create_mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def get_logger(args):
    create_mkdir(args.log_dir)
    log_path = os.path.join(args.log_dir, args.DS+'_'+args.log_file)
    print('logging into %s' % log_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info('#' * 20)

    args_str = ""
    for k, v in sorted(vars(args).items()):
        args_str += "%s" % k + "=" + "%s" % v + "; "
    logger.info(args_str)
    print(args_str)
    logger.info("DS: %s" % args.DS)
    logger.info(f'Split: {args.data_split}, Source Index: {args.source_index}, Target Index: {args.target_index}')

    return logger