import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import torch.nn.functional as F
import math
import random
from PIL import ImageFilter

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return 
    
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.labels))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.labels[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def class_count(data):
    if str(data)[7:16].strip() =='CIFAR10' or str(data)[7:16].strip() =='CIFAR100':
        labels = np.array(data.targets)
    else:
        labels = np.array(data.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num

def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x

def cls_margin(means, center):
    # print('***cls margin', cls)
    if center:
        g_mean = means.mean(dim=0)
        centered_mean = means - g_mean
    else:
        centered_mean = means
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    min_angle = math.degrees(torch.acos(max_cosine).item())
    return max_cosine.item(), min_angle

def M_W_dist(means, W):
    means = means.cpu()
    W = W.cpu()
    g_mean = means.mean(dim=0)
    M_ = means - g_mean
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W / torch.norm(W,'fro')
    return torch.norm(normalized_W - normalized_M)**2, (torch.norm(normalized_W - normalized_M, dim=1)**2).detach().cpu().numpy()

#calculate the avg of cos(v, v') + 1/(C-1)
def coherence(V, C): 
    V = V.cpu()
    G = V @ V.T
    G += torch.ones((C,C)) / (C-1)
    G -= torch.diag(torch.diag(G))
    return torch.norm(G,1).item() / (C*(C-1))

def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    elif str(train_data)[7:16].strip() =='CIFAR10' or str(train_data)[7:16].strip() =='CIFAR100':
        training_labels = np.array(train_data.targets).astype(int)
    else:
        training_labels = np.array(train_data.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        # print(len(preds), len(labels))
        # print(type(preds), type(labels))
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1

def eval_with_preds(train_data, labels, preds, log):
    # Count the number of examples
    # n_total = sum([len(p) for p in preds])
    n_total = len(preds)
    # Split the examples into normal and mixup
    normal_preds, normal_labels = [], []
    mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
    for p, l in zip(preds, labels):
        if isinstance(l, tuple):
            mixup_preds.append(p)
            mixup_labels1.append(l[0])
            mixup_labels2.append(l[1])
            mixup_ws.append(l[2] * np.ones_like(l[0]))
        else:
            normal_preds.append(p)
            normal_labels.append(l)
    # print('normal label:', len(normal_labels), 'normal pred:', len(normal_preds))

    # Calculate normal prediction accuracy
    rsl = {'train_all':0., 'train_many':0., 'train_median':0., 'train_low': 0.}
    if len(normal_preds) > 0:
        # normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
        normal_preds, normal_labels = np.array(normal_preds), np.array(normal_labels)
        n_top1 = mic_acc_cal(normal_preds, normal_labels)
        n_top1_many, \
        n_top1_median, \
        n_top1_low, = shot_acc(normal_preds, normal_labels, train_data)
        rsl['train_all'] += len(normal_preds) / n_total * n_top1
        rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
        rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
        rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

    # Calculate mixup prediction accuracy
    if len(mixup_preds) > 0:
        mixup_preds, mixup_labels, mixup_ws = \
            list(map(np.concatenate, [mixup_preds*2, mixup_labels1+mixup_labels2, mixup_ws]))
        mixup_ws = np.concatenate([mixup_ws, 1-mixup_ws])
        n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
        n_top1_many, \
        n_top1_median, \
        n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, train_data)
        rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
        rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
        rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
        rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

    # Top-1 accuracy and additional string
    # print(rsl['train_median'], rsl['train_low'] )
    print_str = 'Training acc Top1: ' + str(round(rsl['train_all']*100, 3)) + \
                ' Many_top1: ' + str(round(rsl['train_many']*100, 3)) + \
                ' Median_top1: '+ str(round(rsl['train_median']*100, 3)) + \
                ' Low_top1: '+ str(round(rsl['train_low']*100, 3))
    print(print_str)
    if log:
        log.write(print_str)
        log.flush()
    return rsl

def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best, filename):
    
    # filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth', 'best.pth'))


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
