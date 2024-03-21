import argparse
import os
import time
import data_uxnet as data
import sys
from F_swin import SwinUNETR
import numpy as np
import albumentations as al
sys.path.append('/')
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
import random
import warnings

warnings.filterwarnings("ignore")
import logging as log

from visdom import Visdom
import shutil
from models.cnn_res import *
from adan import Adan

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from sklearn import metrics

useCheckpoint = False
my_resume = ''
my_test = 1

# my_lr = 0.00002
# my_batchsize = 256
my_lr = 0.001
my_batchsize = 16

my_epochs = 150

best_acc = 0  # best test accuracy
best_acc_gbt = 0
global mySave_dir
mySave_dir = 'AF_Swin_CA_15'
# visdom_env_name = mySave_dir
start_fold = 1
end_fold = 6

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='res18',
                    help='model')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=my_epochs, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=my_batchsize, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=my_lr, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='10', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default=my_resume, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--save-dir', default=my_save_dir, type=str, metavar='SAVE',# res18
#                     help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=my_test, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

# adan
parser.add_argument('--max-grad-norm', type=float, default=0.0,
                    help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
parser.add_argument('--weight-decay', type=float, default=0.02,
                    help='weight decay, similar one used in AdamW (default: 0.02)')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
parser.add_argument('--opt-betas', default=(0.98, 0.92, 0.99), type=float, nargs='+', metavar='BETA',
                    help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
parser.add_argument('--no-prox', action='store_true', default=False,
                    help='whether perform weight decay like AdamW (default=False)')



# set_seed(2022)
def main():
    global args
    global start_fold
    global end_fold
    global mySave_dir
    args = parser.parse_args()
    torch.cuda.set_device(0)

    results_dir = '/home/zhangconghao/VUnet_1/results/' + mySave_dir
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    save_dir = results_dir + '/code'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # for training, copy files
    cur_dir = '/home/zhangconghao/VUnet_1/'
    pyfiles = [f for f in os.listdir(cur_dir) if f.endswith('.py')]
    for f in pyfiles:
        shutil.copy(os.path.join(cur_dir, f),os.path.join(save_dir,f))
            
    def get_lr(epoch):
        lr = args.lr
        d = 1
        step_num = 1
        warmup_steps = 4000
        # return d**(-0.5)*min(step_num, step_num*warmup_steps**(-1.5))
        
        # if (epoch + 1) > 40:
        #     lr = 0.4*args.lr
        # if (epoch+1) > 80:
        #     lr = 0.5*args.lr
        # if (epoch+1) > 120:
        #     lr = 0.1*args.lr

        return lr

    for fold in range(start_fold, end_fold):

        global best_acc
        best_acc = 0
        swin_unet = SwinUNETR(img_size=32,
                      in_channels=12,
                      out_channels=8,
                      feature_size=12,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=False,
                    )
        swin_unet.cuda()


        # 大部分情况下，设置这个flag可以让内置的cuDNN的auto-tunner
        # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        # 一般来讲，应遵循以下准则：
        # 1. 如果网络输入的数据维度或类型变化不大，设置torch.backbends.cudnn.benchmark = True 可以增加运行效率。
        # 2. 如果网络输入的数据在每次iteration都变化的话，会导致cnDNN每次都寻找一遍最优配置，这样反而会降低运行效率。
        cudnn.benchmark = True

        # swin_unet = DataParallel(swin_unet)

        loss_focal = nn.CrossEntropyLoss()

        # loss_focal = FocalLoss(gamma=0.5)
        # loss_CE = nn.CrossEntropyLoss()
        loss_focal = loss_focal.cuda()

        # optimizer_linear = Adan(linear_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.opt_betas, eps = args.opt_eps, max_grad_norm=args.max_grad_norm, no_prox=args.no_prox)
        optimizer_swin = torch.optim.Adam(
            swin_unet.parameters(),
            args.lr,
            betas=(0.5,0.99),
            weight_decay = args.weight_decay)

        dataset = data.LunaClassifier(
            fold,
            phase = 'train')
        train_loader = DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = args.workers,
            pin_memory=True)
        dataset = data.LunaClassifier(
            fold,
            phase='val')
        val_loader = DataLoader(
            dataset,
            batch_size=my_batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
        logger_name = '/home/zhangconghao/VUnet_1/results/' + mySave_dir +'/log-' + str(fold)
        logging = log.getLogger(logger_name)
        logging.setLevel(log.INFO)
        fileHandler = log.FileHandler(logger_name)
        logging.addHandler(fileHandler)
        for epoch in range(0, args.epochs + 1):
            train(train_loader, swin_unet, loss_focal, epoch, optimizer_swin, logging, get_lr)
            Evaluate('val', val_loader, swin_unet, loss_focal, epoch, mySave_dir, str(fold), logging)
            # test_EBAM(val_loader, swin_unet)


def train(data_loader, swin_net, loss_focal, epoch, optimizer_swin, logging, get_lr):
    start_time = time.time()
    lr = get_lr(epoch)
    swin_net.train()
    for param_group in optimizer_swin.param_groups:
        param_group['lr'] = lr
    train_loss_focal = 0
    train_loss_center = 0
    correct = 0
    total = 0
    # linear_net.train()
    
    swin_net.train()

    # extra_al_0 = al.Compose(
    #     [al.RandomBrightnessContrast(p=0.2)]
    # )
    # extra_al_1 = al.Compose(
    #     [al.GaussNoise(p=0.2)]
    # )

    for i, (img, _, tars) in enumerate(data_loader):
        # img0 = extra_al_0(image=np.array(img))['image']
        # img0 = torch.from_numpy(img0)
        # img1 = extra_al_1(image=np.array(img))['image']
        # img1 = torch.from_numpy(img1)

        # torch.rot90(img0, 0, (2,3))
        # torch.rot90(img1, 2, (2,3))
        
        # imgs = torch.stack([img0, img1], 0).view(-1, 1, 32,32,32)
        # tars = torch.stack([tars*2 + k for k in range(2)], 1).view(-1)
        
        img = img.cuda(non_blocking=True)
        tars = tars.cuda(non_blocking=True)
        feats, output = swin_net(img)
        loss_fc = loss_focal(output, tars)
        loss = loss_fc
        optimizer_swin.zero_grad()
        loss.backward()
        optimizer_swin.step()
        train_loss_focal += loss_fc.data.item()
        
        _, predicted = torch.max(output.data, 1)
        correct += predicted.eq(tars.data).cpu().sum()
        total += tars.size(0)
    end_time = time.time()
    spend_time = end_time - start_time
    # 要增加， acc， recall...

    acc = 100. * correct.data.item()/float(total)
    # print('Epochs:{:>4d} \n   Acc:{:>6.2f}, loss1:{:>6.5f}, loss2:{:>6.5f}, lr:{:.2e}, time:{:>5.2f}'.format(
    #     epoch,
    #     acc,
    #     train_loss1/total,
    #     train_loss2/total,
    #     lr,
    #     spend_time))
    logging.info('\n    Epochs:{:>4d} \n       Acc:{:>6.2f}, loss1:{:>6.5f}, lr:{:.2e}, time:{:>5.2f}'.format(
        epoch,
        acc,
        train_loss_focal/total,
        lr,
        spend_time))

def Evaluate(phase, data_loader, swin_net, loss_focal, epoch, exp_name, fold, logging):
    start_time = time.time()
    if phase != 'test':
        savemodel_path = os.path.join('/home/zhangconghao/VUnet_1/results',exp_name)
        if not os.path.isdir(savemodel_path):
            os.mkdir(savemodel_path)
        fold_name = 'fold_'+fold
        savemodelpath = os.path.join(savemodel_path,fold_name)
        
    swin_net.eval()
    global best_acc
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        TP = FP = FN = TN = 0
        error_idx = []
        for batch_idx, (inputs, _, targets) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(non_blocking = True))
            targets = Variable(targets.cuda(non_blocking = True)).to(torch.long)
            feats, outputs = swin_net(inputs)
            loss_output = loss_focal(outputs, targets)
            test_loss += loss_output.data.item()
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            # error_idx.append(idxs[predicted.eq(targets.data).cpu()])
            TP += ((predicted == 1) & (targets.data == 1)).cpu().sum()
            TN += ((predicted == 0) & (targets.data == 0)).cpu().sum()
            FN += ((predicted == 0) & (targets.data == 1)).cpu().sum()
            FP += ((predicted == 1) & (targets.data == 0)).cpu().sum()

        acc = 100. * correct.data.item() / total
        if phase != 'test':
            if acc > best_acc:
                logging.info('Saving..')
                state = {
                    'net': swin_net,
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, savemodelpath + 'ckpt.t7')
                best_acc = acc
            state = {
                'net': swin_net,
                'acc': acc,
                'epoch': epoch,
            }
            # if epoch % 200 == 0 and epoch != 0:
            #     torch.save(state, savemodelpath + 'ckpt' + str(epoch) + '.t7')
        # best_acc = acc
        tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
        tnr = 100. * TN.data.item() / (TN.data.item() + FP.data.item())
        try:
            precision = 100. * TP.data.item() / (TP.data.item() + FP.data.item())
        except (ZeroDivisionError):
            precision = 0.
        try:
            f1 = 2*tpr*precision/(tpr+precision)
        except (ZeroDivisionError):
            f1 = 0.
        end_time = time.time()
        spend_time = end_time - start_time

        logging.info('\n     Test:\n        Acc:{:>6.2f}, BestAcc:{:>6.2f}, loss:{:>6.5f}, Tpr:{:>6.2f}, Tnr:{:>6.2f}, Precision:{:>6.2f}, F1:{:>6.2f}, time:{:>5.2f}, CE:{:>6.5f}'.format(
            acc,
            best_acc,
            test_loss/total,
            tpr,
            tnr,
            precision,
            f1,
            spend_time, loss_output/total))
        if epoch == args.epochs:
            best_acc = 0
        
if __name__ == '__main__':
    main()