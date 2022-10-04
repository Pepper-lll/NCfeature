import argparse
import collections
import os 
import subprocess

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DistributedSampler

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer

# For num_experts with same settings, we don't want this to set to True.
# This is strongly discouraged because it's misleading: setting it to true does not make it reproducible acorss machine/pytorch version. In addition, it also makes training slower. Use with caution.
deterministic = False
if deterministic:
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # get function handles of loss and metrics
    loss_class = getattr(module_loss, config["loss"]["type"])
    if hasattr(loss_class, "require_num_experts") and loss_class.require_num_experts:
        criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list, num_experts=config["arch"]["args"]["num_experts"])
    elif loss_class.__name__ == 'RIDELossWithNC':
        assert hasattr(model.backbone, "_feat_dim"), "model must have attribute _feat_dim"
        criterion = config.init_obj('loss', module_loss, feat_dim=model.backbone._feat_dim, 
                                    cls_num_list=data_loader.cls_num_list)
    else:
        criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())

    if "type" in config._config["lr_scheduler"]:
        if config["lr_scheduler"]["type"] == "CustomLR":
            lr_scheduler_args = config["lr_scheduler"]["args"]
            gamma = lr_scheduler_args["gamma"] if "gamma" in lr_scheduler_args else 0.1
            if dist.get_rank() == 0:
                print("Scheduler step1, step2, warmup_epoch, gamma:", (lr_scheduler_args["step1"], lr_scheduler_args["step2"], lr_scheduler_args["warmup_epoch"], gamma))
            def lr_lambda(epoch):
                if epoch >= lr_scheduler_args["step2"]:
                    lr = gamma * gamma
                elif epoch >= lr_scheduler_args["step1"]:
                    lr = gamma
                else:
                    lr = 1

                """Warmup"""
                warmup_epoch = lr_scheduler_args["warmup_epoch"]
                if epoch < warmup_epoch:
                    lr = lr * float(1 + epoch) / warmup_epoch
                return lr
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    
def setup_dist(args, port=None, backend="nccl", verbose=False):
    if dist.is_initialized():
        return
    if args.slurm:
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass # use MASTER_PORT in the environment variable
        else:
            os.environ["MASTER_PORT"] = "29500"
        # specify master address
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
        os.environ["RANK"] = str(proc_id)
        
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # if the OS is Windows or macOS, use gloo instead of nccl
        dist.init_process_group(backend=backend)
        # set distributed device
        device = torch.device("cuda:{}".format(local_rank))
        torch.cuda.set_device(device)
        if verbose:
            print("Using device: {}".format(device))
            print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
        return rank, local_rank, world_size, device
    else:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            if verbose:
                print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1
        local_rank = args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)
        return rank, local_rank, world_size, device

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--slurm', action='store_true', default=False,
                      help='enable slurm ddp running')
    # distributed training
    args.add_argument("--local_rank", type=int, 
                      help='local rank for DistributedDataParallel')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--step1'], type=int, target='lr_scheduler;args;step1'),
        CustomArgs(['--step2'], type=int, target='lr_scheduler;args;step2'),
        CustomArgs(['--warmup'], type=int, target='lr_scheduler;args;warmup_epoch'),
        CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
        CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
        CustomArgs(['--reduce_dimension'], type=int, target='arch;args;reduce_dimension'),
        CustomArgs(['--layer2_dimension'], type=int, target='arch;args;layer2_output_dim'),
        CustomArgs(['--layer3_dimension'], type=int, target='arch;args;layer3_output_dim'),
        CustomArgs(['--layer4_dimension'], type=int, target='arch;args;layer4_output_dim'),
        CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts'),
        CustomArgs(['--distribution_aware_diversity_factor'], type=float, target='loss;args;additional_diversity_factor'),
        CustomArgs(['--pos_weight'], type=float, target='arch;args;pos_weight'),
        CustomArgs(['--collaborative_loss'], type=int, target='loss;args;collaborative_loss'),
        CustomArgs(['--distill_checkpoint'], type=str, target='distill_checkpoint')
    ]
    config = ConfigParser.from_args(args, options)
    rank, local_rank, world_size, device = setup_dist(args.parse_args(), verbose=True)
    setattr(config, 'local_rank', local_rank)
    main(config)
