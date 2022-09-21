import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
# import models
from models import my_resnet
import torch.nn.functional as F
import argparse
import sys
import warnings
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from engine import train, train_cls, validate
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
from scipy.sparse.linalg import svds
from randaugment import rand_augment_transform
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

data_root_dict = {'ImageNet_LT': '/home/lxt/data/imagenet',
                  'ImageNet': '/home/lxt/data/imagenet',
                  'iNaturalist18': '/home/lxt/data/iNaturalist18',
                #   'Places': '/nas/dataset/others/places365/',
                  'cifar10': '/home/lxt/data/cifar10',
                  'cifar100': '/home/lxt/data/cifar100',
                  'cifar10_LT': '/home/lxt/data/cifar10',
                  'cifar100_LT': '/home/lxt/data/cifar100',}

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='ImageNet_LT', choices=['cifar10_LT', 'cifar100_LT', 'iNaturalist18', 'ImageNet_LT',\
                    'cifar10', 'cifar100'],
                    help='dataset setting')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnext50_32x4d',
                    choices=['resnet32', 'resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'],
                    help='model architecture:')
parser.add_argument('--loss_type', default="CE", type=str, choices=['CE', 'MSE', 'LDAM'], help='loss type')
parser.add_argument('--loss2', default="NC1Loss", type=str, choices=['NC1Loss', 'CenterLoss'],help='loss type')
parser.add_argument('--NC2Loss', default="v0", type=str, choices=['v0', 'v1', 'v2'],help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, choices=['None', 'Resample', 'DRW', 'Reweight'], help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cls_epochs', default=10, type=int, metavar='N2',
                    help='number of epochs to train classifier')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_schedule', default='cosine', type=str, choices = ['step', 'cosine']
                    , help='initial learning rate')
parser.add_argument('--lamda1', default=0.01, type=float,
                    metavar='L1', help='lamda of feature loss1')
parser.add_argument('--lamda2', default=0.1, type=float,
                    metavar='L2', help='lamda of feature loss2')
parser.add_argument('--lamda3', default=0., type=float,
                    metavar='L3', help='lamda of zero center regularization')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--cls_wd', default=5e-4, type=float,
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
parser.add_argument('--s_aug',default=False, action='store_true', help='use strengthen augmentation')

best_acc1 = 0

def main():
    args = parser.parse_args()
    # prepare_folders(args)
    if args.output:
        args.output_dir = '_'.join(
            [args.dataset, args.arch, 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'l1', str(args.lamda1),
            'l2', str(args.lamda2), 'lr', str(args.lr)])
    output_dir = Path(args.output_dir)
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

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(time_stamp)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        # create model
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

    model = my_resnet.__dict__[args.arch](num_classes=args.num_classes, use_norm=use_norm, ETF_fc=args.ETF_cls)
    ft_dim = model.fc.in_features
    print('Feature dimension:', ft_dim)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    optimizer_model = torch.optim.SGD([{'params':model.parameters(),'lr':args.lr}], 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lr_schedule =='step':
        scheduler = lr_scheduler.MultiStepLR(optimizer_model, milestones=[115, 175], gamma=0.1)  
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
    if args.lamda1 > 0.:
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
            idx = epoch // 70
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
        if args.lamda1 == 0.:
            print('==========> Not add feature loss <==========')
            train_loss, sup_loss, ft_loss1, ft_loss2, max_cos, train_targets, train_preds, train_means, batch_means = \
                train(train_loader, model, criterion_sup, criterion_ft1,\
                optimizer_model, None, epoch, log_training, tf_writer, args)
        elif args.start_ft > 0 and epoch < args.start_ft:
            print('==========> Not add feature loss <==========')
            train_loss, sup_loss, ft_loss1, ft_loss2, max_cos, train_targets, train_preds, train_means, batch_means = \
                train(train_loader, model, criterion_sup, criterion_ft1,\
                optimizer_model, None, epoch, log_training, tf_writer, args)
        else:
            train_loss, sup_loss, ft_loss1, ft_loss2, max_cos, train_targets, train_preds, train_means, batch_means = \
                train(train_loader, model, criterion_sup, criterion_ft1,\
                optimizer_model, optimizer_ftloss, epoch, log_training, tf_writer, args)
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
        # if args.output_dir:
        #     log_testing.write(valid_str + '\n')
        #     log_testing.flush()
        print(valid_str)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_epoch = 0
        if is_best:
            best_epoch = epoch
            print('best epoch:', best_epoch)
            state_dict_pre = model.state_dict()
        best_acc1 = max(acc1, best_acc1)
        if args.gpu is not None:
            W = model.fc.weight
        else:
            W = model.module.fc.weight
        print('feature means avg norm:', torch.mean(torch.norm(train_means, dim=1)))
        M_ = train_means - torch.mean(train_means, dim=0, keepdim=True)
        M_norms = torch.norm(M_, dim=1, keepdim=True)
        MW_dist, MW_dist_cls = M_W_dist(train_means, W)
        print('means and classifier duality:',MW_dist.item())
        mean_cos, mean_angle = cls_margin(train_means, True)
        mean_angle_batch = cls_margin(batch_means.detach(), True)[1]
        weight_cos, weight_angle = cls_margin(W.detach(), False)
        print('maximal cosine, minimum angle between all train feature means:', mean_cos, mean_angle)
        # print('minimum angle between updated feature means:',mean_angle_batch)
        print('maximal cosine, minimum angle between train classifiers:', weight_cos, weight_angle)
        # print('M size:', M_.size())
        cos_M_ = coherence(M_/M_norms, args.num_classes)
        cos_W = coherence(W/torch.norm(W, dim=1, keepdim=True), args.num_classes)
        print('Avg|Cos + 1/(C-1)|', '*M*', cos_M_, '*W*', cos_W)
    
        if tf_writer:
            tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
            tf_writer.add_scalars('analyze/NC2_min_angles', {'total_mean': mean_angle, \
                    'updated_mean': mean_angle_batch, 'weight': weight_angle}, epoch)
            tf_writer.add_scalar('analyze/NC2_Avg|CosM+1(C-1)^-1|', cos_M_, epoch)
            tf_writer.add_scalar('analyze/NC3_dual_mean-classifier', MW_dist, epoch)
            # tf_writer.add_scalars('analyze/NC3_dual_mean-classifier_per_cls', {str(i):x for i, x in enumerate(MW_dist_cls)}, epoch)
        output_best = 'Best Prec@1: %.3f' % (best_acc1) + ' * at Epoch: ' + str(best_epoch)
        print(output_best)
        # if args.output_dir:
        #     log_testing.write(output_best + '\n')
        #     log_testing.flush()
            
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
          
    if args.CRT:
        print("====> Start re-training classifier")
        # num_classes = 100 if args.dataset == 'cifar100' else 10
        use_norm = True if args.loss_type == 'LDAM' else False
        model = my_resnet.__dict__[args.arch](num_classes=args.num_classes, use_norm=use_norm)
        
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        model.load_state_dict(state_dict_pre, strict=False)
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        if use_norm:
            model.fc = nn.Linear(ft_dim, args.num_classes).cuda()
        if args.gpu is not None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
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
        if args.lr_schedule == 'cosine':
            cls_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.cls_epochs, eta_min=0)
        elif args.lr_schedule == 'step':
            cls_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4, 6], gamma=0.1)

        cudnn.benchmark = True
        if args.CB_cls:
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            # train_sampler = ImbalancedDatasetSampler(train_dataset)
            class_counts = np.array(class_count(train_dataset))
            class_weights = 1. / class_counts
            # print("class weights: ", class_weights)
            sample_weights = [0] * len(train_dataset)
            for idx, (data, label, index) in enumerate(train_dataset):
                class_weight = class_weights[label]
                sample_weights[idx] = class_weight
            train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
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
            if args.gpu is not None:
                W = model.fc.weight
            else:
                W = model.module.fc.weight
            MW_dist, MW_dist_cls = M_W_dist(train_means, W)
            weight_angle = cls_margin(W.detach(), False)[1]
            print('means and classifier duality:',MW_dist.item())
            print('minimum angle between train classifiers:',weight_angle)
            acc1, val_loss, valid_targets, valid_preds = validate(val_loader, model, train_means, criterion, None, epoch, log_testing, tf_writer, args, cls=True)
            valid_accs = shot_acc(valid_preds, valid_targets, train_dataset)
            valid_str = 'valid: Many shot - '+str(round(valid_accs[0]*100, 2))+\
                        '; Median shot - '+str(round(valid_accs[1]*100, 2))+ \
                        '; Low shot - '+str(round(valid_accs[2]*100, 2))
            # if args.output_dir:
            #     log_testing.write(valid_str + '\n')
            #     log_testing.flush()
            print(valid_str)
            best_acc_new = max(acc1, best_acc_new)
            is_best = acc1 > best_acc_new
            best_epoch = 0
            if is_best:
                best_epoch = epoch
                print('best epoch:', best_epoch)
            output_best = 'Best Prec@1: %.3f' % (best_acc_new) + ' * at Epoch: ' + str(best_epoch)
            print(output_best)
            if tf_writer:
                tf_writer.add_scalar('acc/test_top1_best', best_acc_new, epoch + args.epochs)
                tf_writer.add_scalars('analyze/NC2_min_angles', {'total_mean': mean_angle, \
                        'updated_mean': mean_angle_batch, 'weight': weight_angle}, epoch + args.epochs)
                tf_writer.add_scalar('analyze/NC3_dual_mean-classifier', MW_dist, epoch + args.epochs)
                # tf_writer.add_scalars('analyze/NC3_dual_mean-classifier_per_cls', {str(i):x for i, x in enumerate(MW_dist_cls)}, epoch + args.epochs)

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
            sanity_check(model.state_dict(), state_dict_pre)
        
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