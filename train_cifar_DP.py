import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7,8'
import builtins
import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
# import models
from models import my_resnet, cifar_resnet, resnet18_cifar
import torch.nn.functional as F
import argparse
import sys
import warnings
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
# from engine import train, train_cls, validate
from engine_cifar import train, train_cls, validate
from data.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import LDAMLoss, FocalLoss
from pathlib import Path
import pickle
from torch.optim import lr_scheduler
# from engine import feature_loss, feature_loss_v2
from losses import NC1Loss, NC1Loss_1, CenterLoss, NC2Loss, NC2Loss_v1, NC2Loss_v2, zero_center, compute_adjustment
# from analyze_collapse import neural_collapse_embedding
from collapse_analysis import analysisNC, MSE_decom
from data import dataloader
from data.ClassAwareSampler import ClassAwareSampler
from scipy.sparse.linalg import svds
from randaugment import rand_augment_transform
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

data_root_dict = {'ImageNet_LT': '/home/lxt/data/imagenet',
                  'ImageNet': '/home/lxt/data/imagenet',
                  'iNaturalist18': '/home/lxt/data/iNaturalist18', 
                #   'Places': '/nas/dataset/others/places365/',
                  'cifar10': '/home/lxt/data/cifar10',
                  'cifar100': '/home/lxt/data/cifar100',
                  'cifar10_LT': '/import/home/xliude/dataset/cifar',
                  'cifar100_LT': '/import/home/xliude/dataset/cifar',}

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10_LT', choices=['cifar10_LT', 'cifar100_LT', 'iNaturalist18', 'ImageNet_LT',\
                    'ImageNet', 'cifar10', 'cifar100'],
                    help='dataset setting')
parser.add_argument('--num_classes', default=10, type=int,
                    help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=['resnet32', 'resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'],
                    help='model architecture:')
parser.add_argument('--loss_type', default="CE", type=str, choices=['CE', 'MSE', 'LDAM'], help='loss type')
parser.add_argument('--loss2', default="NC1Loss", type=str, choices=['NC1Loss', 'CenterLoss'],help='loss type')
parser.add_argument('--NC2Loss', default="v1", type=str, choices=['v0', 'v1', 'v2'],help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='DRW', type=str, choices=['None', 'Resample', 'DRW', 'Reweight'], help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cls_epochs', default=50, type=int, metavar='N2',
                    help='number of epochs to train classifier')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_schedule', default='step', type=str, choices = ['step', 'cosine']
                    , help='initial learning rate')
parser.add_argument('--lr_schedule_cls', default='step', type=str, choices = ['step', 'cosine']
                    , help='initial learning rate')
parser.add_argument('--lamda1', default=0.01, type=float,
                    metavar='L1', help='lamda of feature loss1')
parser.add_argument('--lamda2', default=0.1, type=float,
                    metavar='L2', help='lamda of feature loss2')
parser.add_argument('--lamda3', default=0., type=float,
                    metavar='L3', help='lamda of zero center regularization')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'], 
                    help='optimizer')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--cls_wd', default=0., type=float,
                    help='classifier weight decay (default: 0.1 follow )')
parser.add_argument('--alpha', default=0.5, type=float, metavar='A',
                    help='learning rate for center (class means)')
parser.add_argument('--start_ft', default=0, type=int, 
                    help='starting epoch for fix classifier, and add feature loss')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument ('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--output',default=False, action='store_true')
parser.add_argument('--analyze',default=False, action='store_true', help='analyze collapse')
parser.add_argument('--ETF_cls',default=False, action='store_true', help='fix classifier as simplex-ETF')
parser.add_argument('--norm_cls',default=False, action='store_true', help='normalize the trained balance classifier')
parser.add_argument('--mlp_cls',default=False, action='store_true', help='mlp classifier')
parser.add_argument('--CRT',default=False, action='store_true', help='fix the feature extractor, re-train the classifier')
parser.add_argument('--enlarge_lamda1',default=False, action='store_true', help='enlarge lamda1 at epoch 100')
parser.add_argument('--logit_adj_post', help='adjust logits post hoc', default=False, action='store_true')
parser.add_argument('--tro_post_range', help='check diffrent val of tro in post hoc', type=list,
                    default=[0.25, 0.5, 0.75, 1, 1.5, 2])
parser.add_argument('--logit_adj_train', help='adjust logits in trainingc', default=False, action='store_true')
parser.add_argument('--tro_train', default=1.0, type=float, help='tro for logit adj train')
parser.add_argument('--CB_cls',default=False, action='store_true', help='use class balanced loss to retrain the classifier')
parser.add_argument('--CBS',default=False, action='store_true', help='use class balanced sampler to retrain the classifier')
parser.add_argument('--s_aug',default=False, action='store_true', help='use strengthen augmentation')
parser.add_argument('--mixup',default=False, action='store_true', help='use input mix up')
parser.add_argument('--mixup_alpha', default=0.2, type=float, help='alpha for mix up')
parser.add_argument('--debug',default=False, action='store_true', help='debuging... only train one iteration per epoch')

best_acc1 = 0

def main():
    args = parser.parse_args()
    # prepare_folders(args)
    if args.output:
        if args.logit_adj_train:
            args.output_dir = args.dataset + '/' +str(args.imb_factor)+'/'+ args.arch + '/' + 'logit_adj' + '/'+\
            '_'.join([str(args.train_rule), 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'l1', str(args.lamda1),
            'l2', str(args.lamda2),'l3', str(args.lamda3),'ft', str(args.start_ft), 'lr', str(args.lr), str(args.lr_schedule)])
        elif args.mixup:
            args.output_dir = args.dataset + '/' +str(args.imb_factor)+'/'+ args.arch + '/' + 'mixup' + '/'+\
            '_'.join([str(args.train_rule), 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'l1', str(args.lamda1),
            'l2', str(args.lamda2),'l3', str(args.lamda3),'ft', str(args.start_ft), 'lr', str(args.lr), str(args.lr_schedule)])
        elif args.NC2Loss == 'v1':
            args.output_dir = args.dataset + '/' +str(args.imb_factor)+'/'+ args.arch + '/' + 'NC2v1' + '/'+\
            '_'.join([str(args.train_rule), 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'l1', str(args.lamda1),
            'l2', str(args.lamda2),'l3', str(args.lamda3),'ft', str(args.start_ft), 'lr', str(args.lr), str(args.lr_schedule)])
        else:
            args.output_dir = args.dataset + '/' + str(args.imb_factor) +'/' + args.arch + '/' + \
                '_'.join([str(args.train_rule), 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'l1', str(args.lamda1),
                'l2', str(args.lamda2),'l3', str(args.lamda3),'ft', str(args.start_ft), 'lr', str(args.lr), str(args.lr_schedule)])
        output_dir = Path(args.output_dir)
        while os.path.exists(output_dir):
            args.output_dir += str(1)
            output_dir = Path(args.output_dir)
            print(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        print('print redirected')
        sys.stdout = open(args.output_dir + '/output.txt','wt')
    if args.seed is not None:
        print('=======> Using Fixed Random Seed <========')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    # main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(time_stamp)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        # create model
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == 'cifar100_LT' or args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'cifar10_LT' or args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'iNaturalist18':
        args.num_classes = 8142
    elif args.dataset == 'ImageNet_LT' or args.dataset == 'ImageNet':
        args.num_classes = 1000
    # args.K_A = 50 if args.dataset == 'cifar100' else 5
    use_norm = True if args.loss_type == 'LDAM' else False
    fix_dim = True if args.dataset == 'iNaturalist18' else False
    if args.arch == 'resnet32':
        model = cifar_resnet.__dict__[args.arch](num_classes=args.num_classes, use_norm=use_norm, ETF_fc=args.ETF_cls)
        ft_dim = model.linear.in_features
    elif args.arch == 'resnet18':
        model = resnet18_cifar.ResNet18(num_classes=args.num_classes)
        ft_dim = model.fc.in_features
    print('Feature dimension:', ft_dim)
    # print(model)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = DDP(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    if args.opt == 'sgd':
        optimizer_model = torch.optim.SGD([{'params':model.parameters(),'lr':args.lr}], 
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer_model = torch.optim.Adam([{'params':model.parameters(),'lr':args.lr}],
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        optimizer_model = torch.optim.AdamW([{'params':model.parameters(),'lr':args.lr}],
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        print('uinsupportable optimizer!!!')
    if args.lr_schedule =='step':
        scheduler = lr_scheduler.MultiStepLR(optimizer_model, milestones=[160, 180], gamma=0.1)  
    elif args.lr_schedule == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model, args.epochs, eta_min=0)
    
    cudnn.benchmark = True

    splits = ['train', 'val']
    if args.dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    train_dataset = dataloader.load_data(data_root=data_root_dict[args.dataset],
                                    dataset=args.dataset, phase='train', 
                                    cifar_imb_ratio=args.imb_factor, s_aug=args.s_aug)
    val_dataset = dataloader.load_data(data_root=data_root_dict[args.dataset],
                                    dataset=args.dataset, phase='val', 
                                    cifar_imb_ratio=args.imb_factor)
    if args.dataset == 'ImageNet_LT':
        test_dataset = dataloader.load_data(data_root=data_root_dict[args.dataset],
                                    dataset=args.dataset, phase='test')
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, sampler=None)
    
    train_sampler = None
    per_cls_weights = None 
    if args.train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
        per_cls_weights = None
            
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    
    cls_num_list = class_count(train_dataset)
    args.cls_num_list = cls_num_list

    if args.loss2 == 'NC1Loss':
        criterion_ft1 = NC1Loss(num_classes=args.num_classes, feat_dim=ft_dim)
    elif args.loss2 == 'CenterLoss':
        criterion_ft1 = CenterLoss(num_classes=args.num_classes, feat_dim=ft_dim)
    if args.lamda1 > 0. or args.lamda2 > 0.:
        optimizer_ftloss = torch.optim.SGD(criterion_ft1.parameters(), lr=args.alpha)
        # alpha_scheduler = lr_scheduler.MultiStepLR(optimizer_ftloss, milestones=[100, 160], gamma=0.1)  
    log_training = None
    log_testing = None
    output_dir = Path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    if args.logit_adj_train:
        args.logit_adjustments = compute_adjustment(train_loader, args.tro_train)
    tf_writer = None
    if args.output_dir:
        tf_writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    # print(model.linear.weight.size(), model.linear.weight)
        
    best_acc1 = 0.0
    best_many, best_med, best_few = 0.0, 0.0, 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch < 5:
            lr = args.lr * (epoch+1) / 5
            for param_group in optimizer_model.param_groups:
                param_group['lr'] = lr
        if args.lamda1 > 0.:
            if args.enlarge_lamda1 and epoch == 69:
                args.lamda1 = 0.1
            print('lamda1:', args.lamda1)
            print('alpha:', optimizer_ftloss.param_groups[-1]['lr'])

        if args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // (args.epochs-20)
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        if args.loss_type == 'CE':
            criterion_sup = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'MSE':
            criterion_sup = nn.MSELoss().cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion_sup = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion_sup = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # # train for one epoch
        if args.lamda1 == 0. and args.lamda2 == 0. and args.lamda3 == 0.:
            print('==========> Not add feature loss <==========')
            train_loss, sup_loss, ft_loss1, ft_loss2, max_cos, train_targets, train_preds, M, _ = \
                train(train_loader, model, criterion_sup, criterion_ft1,\
                optimizer_model, None, epoch, log_training, tf_writer, args)
        elif args.start_ft > 0 and epoch < args.start_ft:
            print('==========> Not add feature loss <==========')
            train_loss, sup_loss, ft_loss1, ft_loss2, max_cos, train_targets, train_preds, M, _ = \
                train(train_loader, model, criterion_sup, criterion_ft1,\
                optimizer_model, None, epoch, log_training, tf_writer, args)
        else:
            train_loss, sup_loss, ft_loss1, ft_loss2, max_cos, train_targets, train_preds, M, _ = \
                train(train_loader, model, criterion_sup, criterion_ft1,\
                optimizer_model, optimizer_ftloss, epoch, log_training, tf_writer, args)
        if epoch == args.epochs - 1:
            pickle.dump(M, open(args.output_dir + '/train_M.pkl', 'wb'))
        if epoch >= 5:
            scheduler.step()
        # if args.lamda1 > 0.:
        #     alpha_scheduler.step()
        train_result = eval_with_preds(train_dataset, train_targets, train_preds, log_training)
        
        # evaluate on validation set
        acc1, loss, valid_targets, valid_preds = validate(val_loader, model, criterion_sup, epoch, tf_writer, args)
        valid_accs = shot_acc(valid_preds, valid_targets, train_dataset)
        valid_str = 'valid: Many shot - '+str(round(valid_accs[0]*100, 2))+\
                    '; Median shot - '+str(round(valid_accs[1]*100, 2))+ \
                    '; Low shot - '+str(round(valid_accs[2]*100, 2))
        print(valid_str)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_epoch = 0
        if is_best:
            best_epoch = epoch
            print('best epoch:', best_epoch)
            state_dict_pre = model.state_dict()
        best_acc1 = max(acc1, best_acc1)
        
        if args.dataset=='ImageNet_LT':
            if (epoch+1) >= 90 and (epoch+1) % 10 == 0 :
                print('=====> start to test!!!')
                model.load_state_dict(state_dict_pre, strict=False)
                test_acc1, loss, test_targets, test_preds = validate(test_dataloader, model, criterion_sup, 0, None, args)
                test_accs = shot_acc(test_preds, test_targets, train_dataset)
                test_str = 'valid: Many shot - '+str(round(test_accs[0]*100, 2))+\
                            '; Median shot - '+str(round(test_accs[1]*100, 2))+ \
                            '; Low shot - '+str(round(test_accs[2]*100, 2))
                print(test_str)
                
        if tf_writer:
            tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
            # tf_writer.add_scalars('analyze/NC3_dual_mean-classifier_per_cls', {str(i):x for i, x in enumerate(MW_dist_cls)}, epoch)
        output_best = 'Best Prec@1: %.3f' % (best_acc1) + ' * at Epoch: ' + str(best_epoch)
        print(output_best)
        if output_dir:
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer_model.state_dict(),
            }, is_best=False, filename=output_dir / 'checkpoint_1.pth')
            if is_best:
                save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer_model.state_dict(),
                }, is_best=False, filename=output_dir / 'checkpoint_best.pth')
        state_dict_pre = model.state_dict()  
    if args.dataset=='ImageNet_LT':
        print('=====> start to test!!!')
        model.load_state_dict(state_dict_pre, strict=False)
        test_acc1, loss, test_targets, test_preds = validate(test_dataloader, model, criterion_sup, 0, None, args)
        test_accs = shot_acc(test_preds, test_targets, train_dataset)
        test_str = 'valid: Many shot - '+str(round(test_accs[0]*100, 2))+\
                    '; Median shot - '+str(round(test_accs[1]*100, 2))+ \
                    '; Low shot - '+str(round(test_accs[2]*100, 2))
        print(test_str)
    if args.CRT:
        print("====> Start re-training classifier")
        # num_classes = 100 if args.dataset == 'cifar100' else 10
        use_norm = True if args.loss_type == 'LDAM' else False
        fix_dim = True if args.dataset == 'iNaturalist18' else False

        model = cifar_resnet.__dict__[args.arch](num_classes=args.num_classes, use_norm=use_norm, ETF_fc=args.ETF_cls, fix_dim=fix_dim)
    
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        model.load_state_dict(state_dict_pre, strict=False)
        for name, param in model.named_parameters():
            if 'linear' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        if use_norm:
            model.linear = nn.Linear(ft_dim, args.num_classes).cuda()

        # model.fc.weight.data.normal_(mean=0.0, std=0.01)
        # model.fc.bias.data.zero_()
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.cls_wd)
        if args.lr_schedule_cls == 'cosine':
            cls_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.cls_epochs, eta_min=0)
        elif args.lr_schedule_cls == 'step':
            cls_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

        cudnn.benchmark = True
        if args.CB_cls:
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.CBS:
            # oversample，所有类别数量和最多数类别一致 1280*1000
            # sampler = ClassAwareSampler
            train_sampler = ClassAwareSampler(train_dataset, num_samples_cls=4)
            per_cls_weights = None
        else:
            # oversample minority and down sample majority. 训练样本总数不变
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        if args.loss_type == 'MSE':
            criterion = nn.MSELoss().cuda(args.gpu)
        else:
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        best_acc_new = 0.
        for epoch in range(args.cls_epochs):
            train_loss = train_cls(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer, args)
            cls_scheduler.step()
            acc1, val_loss, valid_targets, valid_preds = validate(val_loader, model, criterion, epoch, tf_writer, args, cls=True)
            valid_accs = shot_acc(valid_preds, valid_targets, train_dataset)
            valid_str = 'valid: Many shot - '+str(round(valid_accs[0]*100, 2))+\
                        '; Median shot - '+str(round(valid_accs[1]*100, 2))+ \
                        '; Low shot - '+str(round(valid_accs[2]*100, 2))
            # if args.output_dir:
            #     log_testing.write(valid_str + '\n')
            #     log_testing.flush()
            print(valid_str)
            is_best = acc1 > best_acc1
            best_epoch = 0
            if is_best:
                best_epoch = epoch
                print('best epoch:', best_epoch)
                state_dict_pre = model.stat_dict()
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f' % (best_acc1) + ' * at Epoch: ' + str(best_epoch)
            print(output_best)
            if tf_writer:
                tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch + args.epochs)
                
            if args.output_dir:
                # log_testing.write(output_best + '\n')
                # log_testing.flush()
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=output_dir / 'checkpoint_cls.pth')
            # sanity_check(model.state_dict(), state_dict_pre)
        if args.dataset=='ImageNet_LT':
            print('=====> start to test!!!')
            model.load_state_dict(state_dict_pre, strict=False)
            test_acc1, loss, test_targets, test_preds = validate(test_dataloader, model, criterion_sup, 0, None, args)
            test_accs = shot_acc(test_preds, test_targets, train_dataset)
            test_str = 'valid: Many shot - '+str(round(test_accs[0]*100, 2))+\
                        '; Median shot - '+str(round(test_accs[1]*100, 2))+ \
                        '; Low shot - '+str(round(test_accs[2]*100, 2))
            print(test_str)
    

    


def sanity_check(state_dict, state_dict_pre):
    """
    Only Linear classifier should change.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    for k in list(state_dict.keys()):
        # only check fc layer
        if 'linear.weight' in k or 'linear.bias' in k or 'fc.weight' in k or 'fc.bias' in k:
            continue
        # name in pretrained model
        # k_pre = 'module.' + k[len('module.'):] \
        #     if k.startswith('module.') else 'module.' + k
        assert ((state_dict[k] == state_dict_pre[k]).all()), \
            '{} is changed in classifier training.'.format(k)
    print("=> sanity check passed.")


if __name__ == '__main__':
    main()