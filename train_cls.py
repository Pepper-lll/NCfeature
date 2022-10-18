import argparse
from email.policy import strict
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import cifar_resnet, my_resnet_original
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from data.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import LDAMLoss, FocalLoss
from pathlib import Path
import pickle
from torch.utils.data import WeightedRandomSampler
from torch.optim import lr_scheduler
from regularizers import MaxNorm_via_PGD
from data import dataloader
from data.ClassAwareSampler import ClassAwareSampler
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
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
parser.add_argument('--dataset', default='ImageNet_LT', help='dataset setting')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='dataset setting')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnext50_32x4d',
                    choices=['resnet32', 'resnet34', 'resnet50', 'resnext50_32x4d'])
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='CBS', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=48, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_factor', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_schedule', default='cosine', type=str, choices = ['step', 'cosine']
                    , help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='/home/lxt/codes/ncGuided/ImageNet_LT/resnext50_32x4d/--DRW_batchsize_128_epochs_200_l1_0.1_l2_1.0_ft_100_lr_0.05_cosine/checkpoint_best.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--output_dir', default=False, action='store_true')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--represent',default=True, action='store_false')
parser.add_argument('--cls',default=False, action='store_true')
parser.add_argument('--MaxNorm',default=False, action='store_true')
parser.add_argument('--s_aug',default=False, action='store_true', help='use strengthen augmentation')

best_acc1 = 0

def main():
    args = parser.parse_args()
    # args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    # prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    log_training = None
    log_testing = None
    tf_writer = None
    if args.output_dir:
        dir = args.resume[:-len('checkpoint_best.pth')]
        print('dir:',dir)
        output_dir = Path(dir)
        print('output_dir:', output_dir)
        print('print redirected')
        sys.stdout = open(dir + '/output_cls.txt','wt')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 8142 if args.dataset == 'iNaturalist18' else 1000
    use_norm = True if args.loss_type == 'LDAM' else False
    model = my_resnet_original.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    

    # load feature extractor from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if 'module' in k:
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
                # if 'linear' in k:
                #     del state_dict[k]
            # print(state_dict.keys())
            model.load_state_dict(state_dict)

            args.start_epoch = 0
            # msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        
    # model.linear = nn.Linear(64, num_classes).cuda()
    
    if args.MaxNorm:
        thresh = 0.1 #threshold value
        pgdFunc = MaxNorm_via_PGD(thresh=thresh)
        pgdFunc.setPerLayerThresh(model) # set per-layer thresholds

    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    # print(model.state_dict()['linear.weight'])
    # print(model.state_dict()['linear.bias'])
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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

    cls_num_list = class_count(train_dataset)
    args.cls_num_list = cls_num_list

    train_sampler = None
    if args.train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
        per_cls_weights = None
    elif args.train_rule == 'CBS':
        train_sampler = ClassAwareSampler(train_dataset, num_samples_cls=4)
        per_cls_weights = None
    if args.dataset == 'ImageNet_LT':
        test_dataset = dataloader.load_data(data_root=data_root_dict[args.dataset],
                                    dataset=args.dataset, phase='test')
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    loss_list = []
    print('====> start classifier retraining')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None 
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'MSE':
            criterion = nn.MSELoss().cuda(args.gpu)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, log_training, args)
        # if scheduler:
        #     scheduler.step()
        loss_list.append(train_loss)
        # evaluate on validation set
        
        acc1, valid_targets, valid_preds = validate(val_loader, model, criterion, epoch, log_testing, args)
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
        best_acc1 = max(acc1, best_acc1)

        
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        if args.output_dir and is_best:
            # Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=output_dir / 'checkpoint_cls_best.pth')
        # sanity_check(model.state_dict(),args.resume)
    print('=====> start to test!!!')
    test_acc1, loss, test_targets, test_preds = validate(test_dataloader, model, criterion, 0, None, args)
    test_accs = shot_acc(test_preds, test_targets, train_dataset)
    test_str = 'valid: Many shot - '+str(round(test_accs[0]*100, 2))+\
                '; Median shot - '+str(round(test_accs[1]*100, 2))+ \
                '; Low shot - '+str(round(test_accs[2]*100, 2))
    print(test_str)
    # if output_dir:
    #     pickle.dump(loss_list, open(output_dir / 'train_cls_loss.pkl', 'wb'))
        

def train(train_loader, model, criterion, optimizer, epoch, log, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # print(target)
        # compute output
        output = model(input)[1]
        # print(output.size(), target.size())
        if str(criterion) == 'MSELoss()':
            loss = criterion(output, F.one_hot(target, num_classes=args.num_classes).float())
        else:
            loss = criterion(output, target)
        # print(loss)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, 
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            if log:
                log.write(output + '\n')
                log.flush()
    return loss

def validate(val_loader, model, criterion, epoch, log, args, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, index) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)[1]
            if str(criterion) == 'MSELoss()':
                loss = criterion(output, F.one_hot(target, num_classes=args.num_classes).float())
            else:
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)

        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        # out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        # print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            # log.write(out_cls_acc + '\n')
            log.flush()

    return top1.avg, np.array(all_targets), np.array(all_preds)

# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     epoch = epoch + 1
#     if epoch <= 5:
#         lr = args.lr * epoch / 5
#     elif epoch > 40:
#         lr = args.lr * 0.001
#     elif epoch > 20:
#         lr = args.lr * 0.01
#     else:
#         lr = args.lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    lr_min = 0
    lr_max = config.lr
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.epochs * 3.1415926535))

    for idx, param_group in enumerate(optimizer.param_groups):
        if idx == 0:
            param_group['lr'] = config.lr_factor * lr
        else:
            param_group['lr'] = 1.00 * lr

def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']
    # print(state_dict.keys())
    # print(state_dict_pre.keys())

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'linear.weight' in k or 'linear.bias' in k:
            continue
        # print()
        # # name in pretrained model
        # k_pre = 'module.encoder_q.' + k[len('module.'):] \
        #     if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

if __name__ == '__main__':
    main()