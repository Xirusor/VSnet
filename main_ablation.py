import argparse
import os
import time
import numpy as np
import data
import sys

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
from swin_unet_mlp import SwinUNETR, SupConLoss, FocalLoss

from adan import Adan

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import cv2
hhh = 0
useCheckpoint = True
my_resume = './results/ablation/FL_Adan_fold_5ckpt.t7'
my_test_fold = hhh
my_test = 1

# my_lr = 0.00002
# my_batchsize = 256
my_lr = 0.0005
my_batchsize = 16

my_epochs = 150

best_acc = 0  # best test accuracy
best_acc_gbt = 0
useVisdom = False
global mySave_dir
mySave_dir = 'ablation'
visdom_env_name = mySave_dir
start_fold = 5
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # set random seed for all gpus
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# set_seed(2022)
def main():
    global args
    global start_fold
    global end_fold
    global mySave_dir
    global my_test_fold
    args = parser.parse_args()
    torch.cuda.set_device(0)

    # 结果保存路径

    # for training, copy files
    # if args.test != 1:
    #     cur_dir = './'
    #     pyfiles = [f for f in os.listdir(cur_dir) if f.endswith('.py')]
    #     for f in pyfiles:
    #         shutil.copy(os.path.join(cur_dir, f), os.path.join(save_dir, f))

    # total = sum([param.nelement() for param in net.parameters()])
    # print(total)

    for fold in range(start_fold, end_fold):
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
        # swin_unet = SwinUNETR(img_size=32,
        #                       in_channels=12,
        #                       out_channels=8,
        #                       feature_size=12,
        #                       drop_rate=0.0,
        #                       attn_drop_rate=0.0,
        #                       dropout_path_rate=0.0,
        #                       use_checkpoint=False,
        #                       )
        global useCheckpoint

        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(my_resume)

        global best_acc
        best_acc = checkpoint['acc']
        # swin_unet = checkpoint['net']
        swin_unet = SwinUNETR(img_size=32,
                              in_channels=12,
                              out_channels=8,
                              feature_size=12,
                              drop_rate=0.0,
                              attn_drop_rate=0.0,
                              dropout_path_rate=0.0,
                              use_checkpoint=False,
                              )
        swin_unet.load_state_dict(checkpoint['net'].state_dict())
        # net_dict = swin_unet.state_dict()
        # new_state_dict = {k: v for k, v in checkpoint_net.named_parameters()}
        # net_dict.update(new_state_dict)
        # swin_unet.load_state_dict(net_dict)
        swin_unet.cuda()


        # 大部分情况下，设置这个flag可以让内置的cuDNN的auto-tunner
        # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        # 一般来讲，应遵循以下准则：
        # 1. 如果网络输入的数据维度或类型变化不大，设置torch.backbends.cudnn.benchmark = True 可以增加运行效率。
        # 2. 如果网络输入的数据在每次iteration都变化的话，会导致cnDNN每次都寻找一遍最优配置，这样反而会降低运行效率。
        cudnn.benchmark = True

        # swin_unet = DataParallel(swin_unet)

        loss_focal = FocalLoss(gamma=0.5)
        # loss_CE = nn.CrossEntropyLoss()
        # loss_focal = loss_CE
        loss_focal = loss_focal.cuda()

        # optimizer_linear = Adan(linear_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.opt_betas, eps = args.opt_eps, max_grad_norm=args.max_grad_norm, no_prox=args.no_prox)
        optimizer_swin = Adan(swin_unet.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.opt_betas,
                              eps=args.opt_eps, max_grad_norm=args.max_grad_norm, no_prox=args.no_prox)

        # optimizer_swin = torch.optim.Adam(
        #     swin_net.parameters(),
        #     args.lr,
        #     betas=(0.5,0.99),
        #     weight_decay = args.weight_decay)

        dataset = data.LunaClassifier(
            fold,
            phase='train')
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
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

        test_EBAM(val_loader, swin_unet)


def test_EBAM(data_loader, net):
    start_time = time.time()
    net.eval()

    correct = 0
    total = 0
    TP = FP = FN = TN = 0

    module_name = []
    features_in_hook = []
    features_out_hook = []

    def hook(module, feature_in, feature_out):
        # global features_in_hook, features_out_hook
        module_name.append(module.__class__)
        features_in_hook.append(feature_in)
        features_out_hook.append(feature_out)
        return None

    feats = targs = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(non_blocking = True))
            targets = Variable(targets.cuda(non_blocking = True)).to(torch.long)
            feature, outputs = net(inputs)

            # net_children = net.children()
            # for child in net_children:
            #     # if not isinstance(child, nn.ReLU6):
            #     child.register_forward_hook(hook=hook)
            # print(targets, idx)
            # for channel in range(8):
            #     feature, outputs = net(inputs)
            #     cam = feature[0,channel,15].cpu().numpy()
            #     cam_img = (cam - cam.min())/(cam.max()-cam.min())
            #     cam_img = np.uint8(255*cam_img)

            #     source_png = np.load('/home/zhangconghao/VUnet_1/data/luna_folds/fold5'+'/' +idx[0] + '_clean.npy')[15]
            #     source_png = (source_png - np.amin(source_png))/(np.amax(source_png)-np.amin(source_png)+1e-5)
            #     source_png = np.uint8(source_png*255)
                
            #     source_png_rgb = cv2.cvtColor(source_png, cv2.COLOR_GRAY2BGR)
                
                
            #     heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

            #     result = heatmap*0.5 + source_png_rgb*0.5

            #     cam_result_path = '/home/zhangconghao/VUnet_1/results/ablation_show' 
                
            #     if targets[0].cpu().tolist():
            #         cv2.imwrite(cam_result_path + '/1/' +  idx[0] + '_' + str(channel)  + '_RGB.jpg', source_png_rgb)
            #         # cv2.imwrite(cam_result_path + '_' + str(channel) + '_Heatmap.jpg', result)
            #         cv2.imwrite(cam_result_path + '/1/' +  idx[0]  + '_' + str(channel) +  '_Result.jpg', result)
            #     else:
            #         cv2.imwrite(cam_result_path + '/0/' +  idx[0]  + '_' + str(channel) + '_RGB.jpg', source_png_rgb)
            #         # cv2.imwrite(cam_result_path + '_' + str(channel) + '_Heatmap.jpg', result)
            #         cv2.imwrite(cam_result_path + '/0/' +  idx[0] + '_' + str(channel) +  '_Result.jpg', result)
 
            total += targets.size(0)
            _, predicted = torch.max(outputs.data, 1)


    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------    
    # TSNE
            
            feats = np.append(feats, feature.cpu().numpy())
            targs = np.append(targs, targets.data.cpu().numpy())
            correct += predicted.eq(targets.data).cpu().sum()
            # error_idx.append(idxs[predicted.eq(targets.data).cpu()])
            TP += ((predicted == 1) & (targets.data == 1)).cpu().sum()
            TN += ((predicted == 0) & (targets.data == 0)).cpu().sum()
            FN += ((predicted == 0) & (targets.data == 1)).cpu().sum()
            FP += ((predicted == 1) & (targets.data == 0)).cpu().sum()
    feats, targs = feats.reshape(-1, 256), targs.reshape(-1, 1)
    from sklearn.metrics import davies_bouldin_score
    
    
    from sklearn.metrics import silhouette_score
    print('silhouette_score', silhouette_score(feats, targs))
    
    # from sklearn.cluster import KMeans
    # from sklearn.metrics import calinski_harabasz_score
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(f)
    # print('calinski_harabasz_score', calinski_harabasz_score(f, kmeans.labels_))
    
    from sklearn.metrics import davies_bouldin_score
    print('DBI', davies_bouldin_score(f, targs))
    
    # from sklearn.metrics import davies_bouldin_score
    # feats, targs = feats.reshape(-1, 256), targs.reshape(-1, 1)
    
    feats, targs = feats.reshape(-1, 256), targs.reshape(-1, 1)
    embeddings_tsne = TSNE()._fit(feats)
    plt.scatter(embeddings_tsne[:,0], embeddings_tsne[:,1], s=150, c=targs, cmap=plt.cm.get_cmap('jet', 2), marker='.', alpha=1.0)
    plt.colorbar(ticks=range(2))
    plt.show()
    # plt.savefig('./results/ablation/CE_TSNE.png')
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------    
    print(correct, total, correct/total)

if __name__ == '__main__':
    main()