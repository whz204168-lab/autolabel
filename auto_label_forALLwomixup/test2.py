import torch
import models.wideresnet as m
import torch.optim as optim
import pickle
import numpy as np
import torch.nn as nn
import dataset.rml2016 as dataset
import torchvision.transforms as transforms
import torch.utils.data as data
from utils import Bar, Logger, AverageMeter, accuracy
import time
from visualise import confusion_matrix, plot_confusion_matrix

with open("RML2016.10a/RML2016.10a_dict.pkl", 'rb') as xd1:  # 这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
    Xd = pickle.load(xd1, encoding='latin1')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
# %%
X_high = []
data1 = []
lblh = []
for i in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21):
    for j in range(i * 10000, (i + 1) * 10000, 1):
        a = X[j]
        b = lbl[j]
        data1.append(a)
        lblh.append(b)
X_high = np.asarray(data1)
# %%
X_low = []
data1 = []
lbll = []
for i in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21):
    for j in range((i - 1) * 10000, i * 10000, 1):
        a = X[j]
        c = lbl[j]
        data1.append(a)
        lbll.append(c)
X_low = np.asarray(data1)

n_examples = X_high.shape[0]
X_high = np.array(X_high).reshape((X_high.shape[0], 2, X_high.shape[2], -1))
lblh1 = list(map(lambda x: mods.index(lblh[x][0]), range(n_examples)))


def load_checkpoint(model, checkpoint, optimizer, loadOptimizer):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['ema_state_dict']
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
        # 如果不需要更新优化器那么设置为false
        if loadOptimizer == True:
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print('loaded! optimizer')
        else:
            print('not loaded optimizer')
    else:
        print('No checkpoint is included')
    return model, optimizer

def create_model(ema=False):
    model = m.WideResNet(num_classes=11)
    model = model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * 0.001

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)

            conf_matrix = np.zeros((len(mods), len(mods)))
            conf_matrix = confusion_matrix(conf_matrix, valloader, model)  # 计算混淆矩阵
            plot_confusion_matrix(conf_matrix, classes=mods, normalize=True, title='Normalized confusion matrix')
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)


use_cuda = torch.cuda.is_available()

model = create_model()
ema_model = create_model(ema=True)
optimizer = optim.Adam(model.parameters())
ema_optimizer= WeightEMA(model, ema_model)
checkpoint = 'result/model_best.pth'

model_, optimizer_ = load_checkpoint(ema_model, checkpoint, optimizer, True)
print("*************")

transform_train = transforms.Compose([
        dataset.RandomPadandCrop(128),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])

train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10(X_high, lblh1, 6600, transform_train=transform_train, transform_val=transform_val)
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
criterion = nn.CrossEntropyLoss()
test_loss, test_acc = validate(test_loader, model_, criterion, 458, use_cuda, mode='Test Stats')
print(test_loss)
print(test_acc)