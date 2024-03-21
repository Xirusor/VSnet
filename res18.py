from asyncore import file_dispatcher
from sklearn import cluster
import torch
from torch import nn
from monai.networks.nets import ViT
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from typing import Tuple, Union
import torch.nn.functional as F
import math
import dill
from torch.nn import Parameter
import numpy as np
from torch.nn.modules.activation import ReLU

import warnings

warnings.filterwarnings('ignore')
class Net(nn.Module):
    def __init__(self):
        
        self.channel_attention = True
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True)
        )

        self.mlp = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 4)
        )
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.myBlock = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size = 3, stride=2, padding = 1),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace = True)
        )

        self.fc1 = nn.Linear(1024, 768)
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 2)

        self.endBlock = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size = 3, stride=2, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 16, kernel_size = 3, stride=2, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 16, kernel_size = 3, stride=2, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
        )

        # self.contrastBlock = nn.Sequential(
        #     nn.Linear(768, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256,128)
        # )
        self.conMlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
        patch_size=(16,16,16)
        # patch_size=(8,8,8)
        self.patch_size=patch_size
        in_channels=24
        out_channels=16
        res_block=True
        conv_block=True
        img_size=(32,32,32)
        feat_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
                    )
        hidden_size=768
        dropout_rate=0.0
        pos_embed='perceptron'
        norm_name='instance'
        mlp_dim=3072
        num_heads=12
        feature_size=16
        self.hidden_size=hidden_size
        self.feat_size = feat_size
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=12,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=True,
            dropout_rate=dropout_rate,
        )

        if patch_size == (8,8,8):
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder3 = UnetrPrUpBlock(
                spatial_dims=3,
                in_channels=hidden_size,
                out_channels=feature_size * 2,
                num_layer=1,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder4 = UnetrPrUpBlock(
                spatial_dims=3,
                in_channels=hidden_size,
                out_channels=feature_size * 4,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.decoder4 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=hidden_size,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
        if patch_size == (16,16,16):
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder2 = UnetrPrUpBlock(
                spatial_dims=3,
                in_channels=hidden_size,
                out_channels=feature_size * 2,
                num_layer=2,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder3 = UnetrPrUpBlock(
                spatial_dims=3,
                in_channels=hidden_size,
                out_channels=feature_size * 4,
                num_layer=1,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder4 = UnetrPrUpBlock(
                spatial_dims=3,
                in_channels=hidden_size,
                out_channels=feature_size * 8,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.decoder5 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=hidden_size,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder4 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):
        x_in = self.preBlock(x)#24
        x_out, cls_token, hidden_states_out = self.vit(x_in)

        x2 = hidden_states_out[3]
        x3 = hidden_states_out[6]
        x4 = hidden_states_out[9]

        if self.channel_attention:
            x_c = torch.cat((x2, x3, x4, x_out), axis=1).reshape(x.shape[0], 4, self.feat_size[0], -1)  # 4*8*768
            c_avg = F.avg_pool2d(x_c, (x_c.size(2), x_c.size(3)), stride=(x_c.size(2), x_c.size(3))).view(x_c.size(0),-1)
            c_max = F.max_pool2d(x_c, (x_c.size(2), x_c.size(3)), stride=(x_c.size(2), x_c.size(3))).view(x_c.size(0),-1)

            c_avg_out = self.mlp(c_avg)
            c_max_out = self.mlp(c_max)

            channel_att_sum = c_avg_out + c_max_out
            scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x_c)
            x_channel = x_c * scale
            x2, x3, x4, x_out = x_channel[:,0,:,:],x_channel[:,1,:,:],x_channel[:,2,:,:],x_channel[:,3,:,:]

        if self.patch_size == (8,8,8):
            enc1 = self.encoder1(x_in)
            x2 = hidden_states_out[6]
            enc2 = self.encoder3(self.proj_feat(x2, self.hidden_size, self.feat_size))
            x3 = hidden_states_out[9]
            enc3 = self.encoder4(self.proj_feat(x3, self.hidden_size, self.feat_size))
            dec3 = self.proj_feat(x_out, self.hidden_size, self.feat_size)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1, enc1)
        else:
            enc1 = self.encoder1(x_in)
            enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
            enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
            enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
            dec4 = self.proj_feat(x_out, self.hidden_size, self.feat_size)
            dec3 = self.decoder5(dec4, enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1, enc1)

        logits1 = self.out(out)
        logits = self.endBlock(logits1)
        logits = logits.view(out.size(0), -1)
        mlp = self.conMlp(logits)
        
        # fc1 = self.fc1(logits)
        # fc1 = fc1 + cls_token
        # fc1 = F.relu(fc1)
        # fc2 = self.fc2(fc1)
        # fc2 = F.relu(fc2)
        # fc3 = self.fc3(fc2)
        # fc3 = torch.sigmoid(fc3)
        return mlp, logits, cls_token

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # pt = torch.sigmoid(predict)
        pt = predict
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 768)
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 2)
    def forward(self, features, cls_token):
        fc1 = self.fc1(features)
        fc1 = fc1 + cls_token
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = F.relu(fc2)
        fc3 = self.fc3(fc2)
        fc3 = torch.sigmoid(fc3)
        return fc2, fc3

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes+1)
    )
    T = torch.FloatTensor(T[:,:-1]).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes=2, sz_embed=128, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss1 = pos_term + neg_term

        return 0.2*loss1

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def get_model():
    net1 = Net()
    net2 = LinearClassifier()
    loss1 = SupConLoss()
    loss2 = Proxy_Anchor()
    return net1, net2, loss1, loss2




from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import cm
import os
def gray_png(input, name='fold_1_0'):
    pd = 15
    file_address = os.path.join(os.getcwd(), 'training/contrast/ablation_result/'+name+'.png')
    tmp1 = input.cpu().numpy()
    tmp2 = np.sum(tmp1, axis=0)[pd,:,:]
    tmp2 = gray_attention(tmp2)
    plt.imsave(file_address, resize(tmp2, [32,32]), cmap=cm.gray)

def gray_attention(input):
    for i in range(len(input)):
        for j in range(len(input[0])):
            if input[i][j]>10:
                input[i][j] = 0
            elif input[i][j]>5:
                input[i][j] = 50
            elif input[i][j]>0:
                input[i][j] = 200
            elif input[i][j]>-5:
                input[i][j] = 200
            else:
                input[i][j] = 200
    return input

def npy2png(file_name, name='source'):
    source = os.path.join(os.getcwd(), 'work/luna_folds/fold1/'+file_name)
    target = os.path.join(os.getcwd(), 'training/contrast/ablation_result/'+name+'.png')
    plt.imsave(target, resize(np.load(source)[15], [32,32]), cmap=cm.gray)
    
    
def cal_DBI(X, labels):
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    #求S
    S = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    Ri = []
    for i in range(n_cluster):
        Rij = []
        #计算Rij
        for j in range(n_cluster):
            if j != i:
                r = (S[i] + S[j]) / euclidean(centroids[i], centroids[j])
                Rij.append(r)
         #求Ri
        Ri.append(max(Rij))
    # 求dbi
    dbi = np.mean(Ri)
    return dbi

# from sklearn.metrics import davies_bouldin_score
# davies_bouldin_score(outputs1.cpu().numpy(), targets.data.cpu().numpy())