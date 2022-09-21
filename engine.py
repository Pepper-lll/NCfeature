import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import *
import time
import torch
from losses import NC1Loss, CenterLoss, NC2Loss, NC2Loss_v1, NC2Loss_v2, zero_center, compute_adjustment
from collapse_analysis import MSE_decom, myNC1
from scipy.sparse.linalg import svds

def train(train_loader, model, criterion_sup, criterion_ft1, \
    optimizer_model, optimizer_ftloss, epoch, log, tf_writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    sup_losses = AverageMeter('Sup_Loss', ':.4e')
    ft_losses1 = AverageMeter('NC1_Feature_Loss', ':.4e')
    ft_losses2 = AverageMeter('NC2_Feature_Loss', ':.4e')
    centers = AverageMeter('center', ':.4e')
    max_coses = AverageMeter('Max_cosine', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    all_targets = []
    all_preds = []
    C = args.num_classes
    mean = [0 for _ in range(C)]
    N = [0 for _ in range(C)]
    dis = [0 for _ in range(C)]
    H = []
    # switch to train mode
    model.train()

    end = time.time()
    if str(criterion_sup) == 'MSELoss()':
        sum_loss = 0.
    for i, (input, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        all_targets.extend(target.cpu().numpy())
        # compute output
        feature, output = model(input)
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        if args.logit_adj_train:
            output += args.logit_adjustments
        if str(criterion_sup) == 'MSELoss()':
            sup_loss = criterion_sup(output, F.one_hot(target, num_classes=args.num_classes).float())
            cri = torch.nn.MSELoss(reduction='sum')
            sum_loss += cri(output, F.one_hot(target, num_classes=args.num_classes).float()).item()
        else:
            sup_loss = criterion_sup(output, target)
        
        # feature = model.forward_embedding(input)
        H.append(feature.detach())
        ft_loss1, c_means = criterion_ft1(feature, target)
        if args.NC2Loss == 'v0': # mean of minimum pair angle
            ft_loss2, max_cos = NC2Loss(c_means)
        elif args.NC2Loss == 'v1': # the one minimum angle
            ft_loss2, max_cos = NC2Loss_v1(c_means)
        elif args.NC2Loss == 'v2': # avg of cosine to -1/(k-1)
            ft_loss2, max_cos = NC2Loss_v2(c_means)
        center_reg = zero_center(c_means)
        loss = sup_loss + args.lamda1 * ft_loss1 + args.lamda2 * ft_loss2 + args.lamda3 * center_reg
        
        for c in range(C):
            # features belonging to class c
            idxs = (target == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0: # If no class-c in this batch
                continue
            h_c = feature[idxs,:] # B CHW
            mean[c] += torch.sum(h_c, dim=0) # CHW
            dis[c] += torch.norm(h_c - c_means[c], dim=1).sum()
            N[c] += h_c.shape[0]
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        sup_losses.update(sup_loss.item(), input.size(0))
        ft_losses1.update(ft_loss1.item(), input.size(0))
        ft_losses2.update(ft_loss2.item(), input.size(0))
        centers.update(center_reg.item(), input.size(0))
        max_coses.update(max_cos.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        if args.lamda1 > 0. and optimizer_ftloss:
            # print('compute gradient of center')
            optimizer_ftloss.zero_grad()
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
            for para in criterion_ft1.parameters():
                para.grad.data *= (1. / args.lamda1)
            optimizer_ftloss.step()
        else:
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            output = ('\n Epoch: [{0}][{1}/{2}], lr: {lr:.6f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Sup_Loss {sup_loss.val:.4f} ({sup_loss.avg:.4f})\t'
                      'NC1_Feature_Loss {ft_loss1.val:.4f} ({ft_loss1.avg:.4f})\t'
                      'NC2_Feature_Loss {ft_loss2.val:.4f} ({ft_loss2.avg:.4f})\t'
                      'Feature_Max_Cosine {max_cos.val:.4f} ({max_cos.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, 
                loss=losses, sup_loss=sup_losses, ft_loss1=ft_losses1, ft_loss2=ft_losses2, center=centers,
                max_cos = max_coses, top1=top1, top5=top5, lr=optimizer_model.param_groups[-1]['lr']))  # TODO
            print(output)
        # break
    if log:
        epoch_out = ('\n ** Epoch: [{0}], lr: {lr:.6f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Sup_Loss {sup_loss.val:.4f} ({sup_loss.avg:.4f})\t'
                      'NC1_Feature_Loss {ft_loss1.val:.4f} ({ft_loss1.avg:.4f})\t'
                      'NC2_Feature_Loss {ft_loss2.val:.4f} ({ft_loss2.avg:.4f})\t'
                      'Feature_Max_Cosine {max_cos.val:.4f} ({max_cos.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, loss=losses, sup_loss=sup_losses, ft_loss1=ft_losses1, ft_loss2=ft_losses2, center=centers,
                max_cos = max_coses, top1=top1, top5=top5, lr=optimizer_model.param_groups[-1]['lr']))  # TODO
        print(epoch_out)
        log.write(epoch_out + '\n')
        log.flush()
    if tf_writer:
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        if args.lamda1 >0.:
            tf_writer.add_scalar('loss_nc1/train', ft_losses1.avg, epoch)
        if args.lamda2 > 0.:
            tf_writer.add_scalar('loss_nc2/train', ft_losses2.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', optimizer_model.param_groups[-1]['lr'], epoch)
    
    for c in range(C):
        if N[c] == 0:
            mean[c] = torch.zeros(feature.size(-1)).cuda()
            continue
        mean[c] /= N[c]
        dis[c] /= N[c]
        dis[c] = round(dis[c].item(), 4)
    M = torch.stack(mean).cpu()  # K * P
    Mnorms = torch.norm(M, dim=1)
    McoV = round((torch.std(Mnorms)/torch.mean(Mnorms)).item(), 4)
    print('CoV of feature means:', McoV)
    avg_mean_norm = torch.mean(torch.norm(M, dim=1))
    print('avg mean norm:', avg_mean_norm.item())
    M_ = M - torch.mean(M, dim=0, keepdim=True)
    avg_centered_mean_norm = torch.mean(torch.norm(M_, dim=1))
    M_norms = torch.norm(M_, dim=1)
    M_coV = round((torch.std(M_norms)/torch.mean(M_norms)).item(), 4)
    print('CoV of centered feature means:', M_coV)
    print('avg centered mean norm', avg_centered_mean_norm.item())
    # print('norm of global mean:', g_norm.item())
    # mean_norm = round(torch.mean(M_norms).item(), 4)
    # print('avg of centered means:', mean_norm)
    if tf_writer:    
        # tf_writer.add_scalars('check/dist_batch-total_means', {str(i):x for i, x in enumerate(M_dist)}, epoch)
        # tf_writer.add_scalars('analyze/NC1_feature_to_mean_dis', {str(i):x for i, x in enumerate(dis)}, epoch)
        tf_writer.add_scalar('analyze/NC2_CoV_of_centered_mean_norm', M_coV, epoch)
        # tf_writer.add_scalars('analyze/NC2_centered_mean_norm', {str(i):x for i, x in enumerate(M_norms.detach().cpu().numpy())}, epoch)
        # tf_writer.add_scalar('analyze/global_mean_norm', g_norm, epoch)
        # tf_writer.add_scalar('analyze/avg_mean_norm', mean_norm, epoch)
    if epoch % 5 == 0 or epoch == args.epochs-1:
        H = torch.cat(H)
        H = H.detach().cpu()
        if args.gpu is not None:
            W = model.fc.weight.detach().cpu()
            b = model.fc.bias.detach().cpu()
        else:
            W = model.module.fc.weight.detach().cpu()
            b = model.module.fc.bias.detach().cpu()
        Y = torch.from_numpy(np.array(all_targets))
        Sw_invSb = myNC1(Y, H, C, M)
        print('tr(Sw/Sb):', Sw_invSb)
        if tf_writer:
            tf_writer.add_scalar('analyze/tr(Sw_invSb)_orig', Sw_invSb, epoch)
        if str(criterion_sup) == 'MSELoss()':
            # print('Y;', Y.size(), 'H:', H.size(), 'W:', W.size())
            L2loss_sum, LNC1, LNC23, Lperp = MSE_decom(Y, H, W, b, C, N, M)
            
            print(L2loss_sum, LNC1, LNC23, Lperp)
            print('~~~~~suplosses total:', sup_losses.avg * sum(N) * C, sum_loss)
            if tf_writer:
                tf_writer.add_scalars('analyze/MSE_decompose', {'loss_sum':L2loss_sum, 'LNC1':LNC1, 'LNC23':LNC23, 'Lperp':Lperp}, epoch)
    return losses.avg, sup_losses.avg, ft_losses1.avg, ft_losses2.avg, \
        max_coses.avg, np.array(all_targets), np.array(all_preds), M, c_means

def train_cls(train_loader, model, criterion, optimizer, epoch, log, tf_writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()

    end = time.time()
    all_targets = []
    all_preds = []
    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # compute output
        all_targets.extend(target.cpu().numpy())
        # compute output
        output = model(input)
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        if str(criterion) == 'MSELoss()':
            loss = criterion(output, F.one_hot(target, num_classes=args.num_classes).float())
        else:
            loss = criterion(output, target)
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
            output = ('\n Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, 
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
        # break
    if log:
        epoch_out = ('\n ** Epoch: [{0}], lr: {lr:.6f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
        print(epoch_out)
        log.write(epoch_out + '\n')
        log.flush()
    if tf_writer:
        tf_writer.add_scalar('loss/train', losses.avg, epoch+args.epochs)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch+args.epochs)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch+args.epochs)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch+args.epochs)
    return loss, np.array(all_targets), np.array(all_preds)


def validate(val_loader, model, criterion_sup, epoch, tf_writer, args, cls=False, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    sup_losses = AverageMeter('Sup_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    C = args.num_classes
    N = [0 for _ in range(C)]
    dis = [0 for _ in range(C)]
    H = []
    mean = [0 for _ in range(C)]
    # all_features = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, index) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            feature, output = model(input)
            if str(criterion_sup) == 'MSELoss()':
                loss = criterion_sup(output, F.one_hot(target, num_classes=args.num_classes).float())
            else:
                loss = criterion_sup(output, target)
            # feature = model.forward_embedding(input)
            H.append(feature.detach())
            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]
                if len(idxs) == 0: # If no class-c in this batch
                    continue
                h_c = feature[idxs,:] # B CHW
                mean[c] += torch.sum(h_c, dim=0)
                N[c] += h_c.shape[0]
            # # all_features.extend(feature.detach().cpu().numpy())

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
            # break
        output = ('Epoch [{epoch}] {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(epoch=epoch, flag=flag, top1=top1, top5=top5, loss=losses))
        # out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        # print(out_cls_acc)
    for c in range(C):
        if N[c] > 0:
            mean[c] /= N[c]
    if epoch == args.epochs - 1:
        M = torch.stack(mean)
        H = torch.cat(H).detach()
        Y = torch.from_numpy(np.array(all_targets)).cuda()
        # print('Y:', Y.size(), 'H:', H.size())
        Sw_invSb = myNC1(Y, H, C, M)
        print('test tr(Sw/Sb):', Sw_invSb)
        if tf_writer:
            tf_writer.add_scalar('analyze_test/tr(Sw_invSb)', Sw_invSb, epoch)
            # tf_writer.add_scalars('analyze_test/NC1_feature_to_center', {str(i):x for i, x in enumerate(dis)}, epoch+args.epochs if cls else epoch)
            tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch+args.epochs if cls else epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch+args.epochs if cls else epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch+args.epochs if cls else epoch)
    return top1.avg, sup_losses.avg, np.array(all_targets), np.array(all_preds)
    #, mean_norm_std, angle_std, large_w, small_w, large_b, small_b
