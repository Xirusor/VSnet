import argparse
from importlib.util import source_hash
import os
import time
import numpy as np
import data
from importlib import import_module
import sys
sys.path.append('../')
import torch
from torch.nn import DataParallel
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import math
import random
import warnings
warnings.filterwarnings("ignore")
import logging as log
# import megengine
from visdom import Visdom
import shutil
from models.cnn_res import *
from swin_unet_mlp import SwinUNETR, SupConLoss, FocalLoss

from adan import Adan
# megengine.dtr.eviction_shreshold = '8GB'
# megengine.dtr.enable()
# 接在 results/save_dir/ 后面的
# my_save_dir = 'res18/test/'

hhh = 0
useCheckpoint = False
my_resume = []
my_test_fold = hhh
my_test = 0

# my_lr = 0.00002
# my_batchsize = 256
my_lr = 0.0005
my_batchsize = 2

my_epochs = 100

best_acc = 0  # best test accuracy
best_acc_gbt = 0
useVisdom = False
global mySave_dir
# mySave_dir = 'FL_BS16_gamma0.5_LR00005_Adan_random1'
# mySave_dir = 'CE_BS16_LR00005_Adam_seed2022'
# mySave_dir = 'CE_BS16_LR00005_Adan_seed2022'
mySave_dir = 'test'

visdom_env_name = mySave_dir
start_fold = 1
end_fold = 6
usePALoss = False

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
parser.add_argument('--test-fold', default=my_test_fold, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

# adan
parser.add_argument('--max-grad-norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
parser.add_argument('--weight-decay', type=float, default=0.02,  help='weight decay, similar one used in AdamW (default: 0.02)')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
parser.add_argument('--opt-betas', default=(0.98, 0.92, 0.99), type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
parser.add_argument('--no-prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # set random seed for all gpus
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(2022)
def main():
    global args
    global start_fold
    global end_fold
    global mySave_dir
    global my_test_fold
    args = parser.parse_args()
    torch.cuda.set_device(0)

    # 结果保存路径

    results_dir = '/home/zhangconghao/VUnet_1/results/' + mySave_dir
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    save_dir = results_dir + '/code'
    start_epoch = args.start_epoch
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # for training, copy files
    if args.test!=1:
        cur_dir = '/home/zhangconghao/VUnet_1/'
        pyfiles = [f for f in os.listdir(cur_dir) if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(os.path.join(cur_dir, f),os.path.join(save_dir,f))

    # total = sum([param.nelement() for param in net.parameters()])
    # print(total)

    def getFreeId():
        import pynvml 

        pynvml.nvmlInit()
        def getFreeRatio(id):
            handle = pynvml.nvmlDeviceGetHandleByIndex(id)
            use = pynvml.nvmlDeviceGetUtilizationRates(handle)
            ratio = 0.5*(float(use.gpu+float(use.memory)))
            return ratio

        deviceCount = pynvml.nvmlDeviceGetCount()
        available = []
        for i in range(deviceCount):
            if getFreeRatio(i)<70:
                available.append(i)
        gpus = ''
        for g in available:
            gpus = gpus+str(g)+','
        gpus = gpus[:-1]
        return gpus

    def setgpu(gpuinput):
        freeids = getFreeId()
        if gpuinput=='all':
            gpus = freeids
        else:
            gpus = gpuinput
            if any([g not in freeids for g in gpus.split(',')]):
                raise ValueError('gpu'+g+'is being used')
        print('using gpu '+gpus)
        os.environ['CUDA_VISIBLE_DEVICES']=gpus
        return len(gpus.split(','))

    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu

    def get_lr(epoch):
        lr = args.lr
        d = 1
        step_num = 1
        warmup_steps = 4000
        # return d**(-0.5)*min(step_num, step_num*warmup_steps**(-1.5))
        
        # if (epoch + 1) > 40:
        #     lr = 0.4*args.lr
        if (epoch+1) > 80:
            lr = 0.5*args.lr
        if (epoch+1) > 150:
            lr = 0.1*args.lr

        return lr


    for fold in range(start_fold,end_fold):
        # if args.resume:
        #     model = import_module(args.model)
        #     net1, net2, loss1, loss2 = model.get_model()
        #     print('==> Resuming from checkpoint..')
        #     checkpoint = torch.load(my_resume)
        #     net_dict = net.state_dict()
        #     new_state_dict = {k: v for k, v in checkpoint.items() if
        #                       k in net_dict and k != 'vit.patch_embedding.position_embeddings'}
        #     net_dict.update(new_state_dict)
        #     net.load_state_dict(net_dict)
        # set_seed(2022)
        # model = import_module(args.model)
        # net1, net2, loss1, loss2 = model.get_model()
        swin_unet = SwinUNETR(img_size=32,
                      in_channels=12,
                      out_channels=8,
                      feature_size=12,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=False,
                    )
        global useCheckpoint
        if useCheckpoint:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(my_resume[0])
            net1 = checkpoint['net1']
            net2 = checkpoint['net2']
            global best_acc
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        # 大部分情况下，设置这个flag可以让内置的cuDNN的auto-tunner
        # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        # 一般来讲，应遵循以下准则：
        # 1. 如果网络输入的数据维度或类型变化不大，设置torch.backbends.cudnn.benchmark = True 可以增加运行效率。
        # 2. 如果网络输入的数据在每次iteration都变化的话，会导致cnDNN每次都寻找一遍最优配置，这样反而会降低运行效率。
        cudnn.benchmark = True

        swin_net = DataParallel(swin_unet)
        
        # loss_focal = FocalLoss(gamma=0.5)
        loss_focal = nn.CrossEntropyLoss()
        loss_focal = loss_focal.cuda()
        
        # optimizer_linear = Adan(linear_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.opt_betas, eps = args.opt_eps, max_grad_norm=args.max_grad_norm, no_prox=args.no_prox)
        optimizer_swin = Adan(swin_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.opt_betas, eps = args.opt_eps, max_grad_norm=args.max_grad_norm, no_prox=args.no_prox)

        # optimizer_swin = torch.optim.Adam(
        #     swin_net.parameters(),
        #     args.lr,
        #     betas=(0.5,0.99),
        #     weight_decay = args.weight_decay)

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
            phase = 'val')
        val_loader = DataLoader(
            dataset,
            batch_size = 8,
            shuffle = False,
            num_workers = args.workers,
            pin_memory=True)

        if args.test:
            test_EBAM(val_loader, net1, net2)
            sys.exit()
        
        logger_name = '/home/zhangconghao/VUnet_1/results/' + mySave_dir +'/log-' + str(fold)
        logging = log.getLogger(logger_name)
        logging.setLevel(log.INFO)
        fileHandler = log.FileHandler(logger_name)
        logging.addHandler(fileHandler)

        global visdom_env_name
        if useVisdom:
            viz = Visdom(env=visdom_env_name)
            step_list = [0]
            train_loss = viz.line(X=np.array([0]), Y=np.array([0.05]), opts={'title':'Fold-'+str(fold)+'_Loss'})
            two_acc = viz.line(X=np.array([0]), Y=np.array([0.0]), opts={'title':'Fold-'+str(fold)+'_Acc'})
            tpr_tnr = viz.line(X=np.array([0]), Y=np.array([0.0]), opts={'title':'Fold-'+str(fold)+'_Test Tpr & Tnr'})
        else:
            viz=None
            step_list=None
            train_loss=None
            two_acc=None
            tpr_tnr=None
        print('Fold: '+str(fold))
        for epoch in range(start_epoch, args.epochs + 1):
            if useVisdom:
                step_list.append(step_list[-1] + 1)

            train(train_loader, swin_net, loss_focal, epoch, optimizer_swin, get_lr, logging, viz, step_list, train_loss, two_acc)
            Evaluate('val', val_loader, swin_net, loss_focal, epoch, mySave_dir, str(fold), logging, viz, step_list, train_loss, two_acc, tpr_tnr)
        print('finished!')


def train(data_loader, swin_net, loss_focal, epoch, optimizer_swin, get_lr, logging, viz, step_list, train_loss_viz, two_acc):
    start_time = time.time()
    lr = get_lr(epoch)
    swin_net.train()
    for param_group in optimizer_swin.param_groups:
        param_group['lr'] = lr
    train_loss_focal = 0
    correct = 0
    total = 0
    # linear_net.train()
    
    swin_net.train()
    global usePALoss

    for i, (img1, _, tars) in enumerate(data_loader):
        img = img1.cuda(non_blocking=True)
        tars = tars.cuda(non_blocking=True)
        output = swin_net(img)
        loss_fc = loss_focal(output, tars)
        optimizer_swin.zero_grad()
        loss_fc.backward()
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

    if useVisdom:
        # viz.line(X=np.array([step_list[-1]]), Y=np.array([train_loss_sc/total]), win=train_loss_viz, update='append', name='contrast Loss')
        viz.line(X=np.array([step_list[-1]]), Y=np.array([train_loss_focal/total]), win=train_loss_viz, update='append', name='classifier Loss')
        viz.line(X=np.array([step_list[-1]]), Y=np.array([acc]), win=two_acc, update='append', name='Train Acc')


def Evaluate(phase, data_loader, swin_net, loss_focal, epoch, exp_name, fold, logging, viz, step_list, train_loss, two_acc, tpr_tnr):
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
        global usePALoss
        error_idx = []
        for batch_idx, (inputs, _, targets) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(non_blocking = True))
            targets = Variable(targets.cuda(non_blocking = True)).to(torch.long)
            # _, outputs, cls_token = net1(inputs)
            outputs = swin_net(inputs)
            # fc, outputs = net2(outputs, cls_token)
            # if usePALoss:
            #     loss_output_pa = loss(fc, targets)
            #     loss_output_ce = loss3(outputs, targets)
            #     loss_output = loss_output_pa+loss_output_ce
            # else:
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
                    'net': swin_net.module,
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, savemodelpath + 'ckpt.t7')
                best_acc = acc
            state = {
                'net': swin_net.module,
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

        # print('Test:\n  Acc:{:>6.2f}, BestAcc:{:>6.2f}, loss:{:>6.5f}, Tpr:{:>6.2f}, Tnr:{:>6.2f}, time:{:>5.2f}'.format(
        #     acc,
        #     best_acc,
        #     test_loss/total,
        #     tpr,
        #     tnr,
        #     spend_time))
        if usePALoss:
            logging.info('\n     Test:\n        Acc:{:>6.2f}, BestAcc:{:>6.2f}, loss:{:>6.5f}, Tpr:{:>6.2f}, Tnr:{:>6.2f}, Precision:{:>6.2f}, F1:{:>6.2f}, time:{:>5.2f}'.format(
                acc,
                best_acc,
                test_loss/total,
                tpr,
                tnr,
                precision,
                f1,
                spend_time))
        else:
            logging.info('\n     Test:\n        Acc:{:>6.2f}, BestAcc:{:>6.2f}, loss:{:>6.5f}, Tpr:{:>6.2f}, Tnr:{:>6.2f}, Precision:{:>6.2f}, F1:{:>6.2f}, time:{:>5.2f}, CE:{:>6.5f}'.format(
                acc,
                best_acc,
                test_loss/total,
                tpr,
                tnr,
                precision,
                f1,
                spend_time, loss_output/total))
        if useVisdom:
            viz.line(X=np.array([step_list[-1]]), Y=np.array([test_loss/total]), win=train_loss, update='append', name='Test Loss')
            viz.line(X=np.array([step_list[-1]]), Y=np.array([acc]), win=two_acc, update='append', name='Acc')
            viz.line(X=np.array([step_list[-1]]), Y=np.array([tpr]), win=tpr_tnr, update='append', name='Tpr')
            viz.line(X=np.array([step_list[-1]]), Y=np.array([tnr]), win=tpr_tnr, update='append', name='Tnr')
            viz.line(X=np.array([step_list[-1]]), Y=np.array([f1]), win=tpr_tnr, update='append', name='F1')

        if epoch == args.epochs:
            best_acc = 0

def test_EBAM(data_loader, net1, net2, loss):
    return None
    # start_time = time.time()
    # net1.eval()
    # net2.eval()
    # with torch.no_grad():
    #     test_loss = 0
    #     correct = 0
    #     total = 0
    #     TP = FP = FN = TN = 0
    #     global usePALoss
    #     loss3 = nn.CrossEntropyLoss()
    #     loss3 = loss3.cuda()
    #     for batch_idx, (inputs, _, targets) in enumerate(data_loader):
    #         # inputs = Variable(inputs.cuda(non_blocking = True))
    #         # targets = Variable(targets.cuda(non_blocking = True)).to(torch.long)
    #         fm, _, outputs1, cls_token = net1(inputs) 
    #         fc, outputs = net2(outputs1, cls_token)
    #         # if usePALoss:
    #         #     loss_output_pa = loss(fc, targets)
    #         #     loss_output_ce = loss3(outputs, targets)
    #         #     loss_output = loss_output_pa+loss_output_ce
    #         # else:
    #         #     loss_output = loss3(outputs, targets)
    #         # test_loss += loss_output.data.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         import cv2
    #         thefold = 1
    #         theidx = -1
    #         cur_dir = '/home/zhangconghao/Unet/work/luna_folds'
    #         for dir_ in os.listdir(cur_dir):
    #             if dir_.split('d')[1] == str(thefold):
    #                 for ct_id in os.listdir(os.path.join(cur_dir, dir_)):
    #                     if ct_id.split('_')[-1] == 'clean.npy':
    #                         ct_id = ct_id.split('_clean.npy')[0]
    #                         theidx += 1
    #                         leibie = targets.data.numpy()[theidx]
    #                         file_name = str(theidx) + '_' + ct_id
                            
    #                         os.makedirs(os.path.join(os.getcwd(), 'training/contrast/ablation_result/'+ str(leibie) + '/' + file_name))
    #                         featuremap = gray_png(fm[theidx], name=file_name + '_heatmap')
    #                         cv2.imwrite('/home/zhangconghao/Unet/training/contrast/ablation_result/' + str(leibie) + '/' + file_name + '/' + 'featuremap.png', featuremap)
    #                         source_png = np.load('/home/zhangconghao/Unet/work/luna_folds/fold'+str(thefold)+'/' +ct_id + '_clean.npy')[15]
    #                         source_png = (source_png - np.amin(source_png))/(np.amax(source_png)-np.amin(source_png)+1e-5)
    #                         source_png = np.round(source_png*255)
    #                         cv2.imwrite('/home/zhangconghao/Unet/training/contrast/ablation_result/' + str(leibie) + '/' + file_name + '/' +file_name +'.png', source_png)
    #                         source_png = cv2.imread('/home/zhangconghao/Unet/training/contrast/ablation_result/' + str(leibie) + '/' + file_name + '/' +file_name+'.png', cv2.IMREAD_GRAYSCALE)
    #                         heatmap = cv2.applyColorMap(featuremap, cv2.COLORMAP_JET)
    #                         source_png1 = source_png[np.newaxis,:]
    #                         source_png2 = np.repeat(source_png1, 3, axis=0)
    #                         source_png3 = np.transpose(source_png2, (1,2,0))
    #                         source_png = cv2.applyColorMap(source_png, cv2.COLORMAP_JET)
    #                         s_img = heatmap*0.4 + source_png
    #                         cv2.imwrite('/home/zhangconghao/Unet/training/contrast/ablation_result/' + str(leibie) + '/'  + file_name + '/' +file_name+'res.png', s_img)

    #         from sklearn.metrics import davies_bouldin_score
    #         print(davies_bouldin_score(fc.cpu().numpy(), targets.data.cpu().numpy()))
    #         total += targets.size(0)
    #         correct += predicted.eq(targets.data).cpu().sum()
    #         TP += ((predicted == 1) & (targets.data == 1)).cpu().sum()
    #         TN += ((predicted == 0) & (targets.data == 0)).cpu().sum()
    #         FN += ((predicted == 0) & (targets.data == 1)).cpu().sum()
    #         FP += ((predicted == 1) & (targets.data == 0)).cpu().sum()

    #     acc = 100. * correct.data.item() / total
    #     # best_acc = acc
    #     tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
    #     tnr = 100. * TN.data.item() / (TN.data.item() + FP.data.item())
    #     try:
    #         precision = 100. * TP.data.item() / (TP.data.item() + FP.data.item())
    #     except (ZeroDivisionError):
    #         precision = 0.
    #     try:
    #         f1 = 2*tpr*precision/(tpr+precision)
    #     except (ZeroDivisionError):
    #         f1 = 0.
    #     end_time = time.time()
    #     spend_time = end_time - start_time

    #     print('Test:\n  Acc:{:>6.2f}, BestAcc:{:>6.2f}, loss:{:>6.5f}, Tpr:{:>6.2f}, Tnr:{:>6.2f}, time:{:>5.2f}'.format(
    #         acc,
    #         best_acc,
    #         test_loss/total,
    #         tpr,
    #         tnr,
    #         spend_time))



def gray_png(input, name='heatmap1'):
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    import cv2
    pd = 15
    file_address = os.path.join(os.getcwd(), 'training/contrast/ablation_result/'+name+'.png')
    tmp1 = input.cpu().numpy()
    tmp2 = np.sum(tmp1, axis=0)[pd,:,:]
    tmp2 = gray_attention(tmp2)
    tmp3 = cv2.applyColorMap(np.array(tmp2, np.uint8), 2)
    return tmp3
    # plt.imsave(file_address, resize(tmp2, [32,32]), cmap=cm.gray)

def gray_attention(input):
    input = (input - np.amin(input))/(np.amax(input)-np.amin(input)+1e-5)
    input = np.round(input*255)
    return input
    
    
    for i in range(len(input)):
        for j in range(len(input[0])):
            if input[i][j]>8:
                input[i][j] = 50
            elif input[i][j]>4:
                input[i][j] = 100
            elif input[i][j]>0:
                input[i][j] = 150
            elif input[i][j]>-5:
                input[i][j] = 200
            else:
                input[i][j] = 200
    return input

def my(input):
    for i in range(len(input)):
        for j in range(len(input[0])):
                input[i][j] = 255 - input[i][j]
    return input

def test(data_loader, net):
    start_time = time.time()
    use_cuda = True
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        TP = FP = FN = TN = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(non_blocking = True))
            targets = Variable(targets.cuda(non_blocking = True)).to(torch.long)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # _, predicted1 = torch.max(outputs[1].data, 1)
            # predicted = outputs
            # predicted[predicted>=0.5] = 1
            # predicted[predicted<0.5] = 0

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            TP += ((predicted == 1) & (targets.data == 1)).cpu().sum()
            TN += ((predicted == 0) & (targets.data == 0)).cpu().sum()
            FN += ((predicted == 0) & (targets.data == 1)).cpu().sum()
            FP += ((predicted == 1) & (targets.data == 0)).cpu().sum()

        acc = 100. * correct.data.item() / total
        tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
        tnr = 100. * TN.data.item() / (TN.data.item() + FP.data.item())
        precision = 100. * TP.data.item() / (TP.data.item() + FP.data.item())
        f1 = 2*tpr*precision/(tpr+precision)
        end_time = time.time()
        spend_time = end_time - start_time

        print('Test:\n  Acc:{:>6.2f},Tpr:{:>6.2f}, Tnr:{:>6.2f}, Precision:{:>6.2f}, F1-socre:{:>6.2f}, time:{:>5.2f}'.format(
            acc,
            tpr,
            tnr,
            precision,
            f1,
            spend_time))


if __name__ == '__main__':
    main()