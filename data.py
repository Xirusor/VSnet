import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
import transforms as transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # set random seed for all gpus
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(2022)


class LunaClassifier(Dataset):
    def __init__(self, fold, phase = 'train',split_comber=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')

        self.fold = fold
        self.phase = phase
        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []
        self.idcs = []

        train_pixvlu,train_npix,train_pixvar = 0,0,0
        test_pixvlu,test_npix,test_pixvar = 0,0,0

        cur_dir = '/home/zhangconghao/VUnet_1/data/luna_folds'
        for dir_ in os.listdir(cur_dir):
            if dir_.split('d')[1] == str(fold):
                subset = dir_.split('d')[1]
                for ct_id in os.listdir(os.path.join(cur_dir, dir_)):
                    if ct_id.split('_')[-1] == 'clean.npy':
                        ct_id = ct_id.split('_clean.npy')[0]
                        # print(ct_id)
                        data_clean = np.load(cur_dir+'/fold'+str(subset)+'/'+str(ct_id)+'_clean.npy')
                        self.test_data.append(data_clean)
                        data_labels = np.load(cur_dir+'/fold'+str(subset)+'/'+str(ct_id)+'_label.npy')
                        self.test_labels.append(int(data_labels[-1]))
                        test_pixvlu += np.sum(data_clean)
                        test_pixvar += np.prod(data_clean*data_clean)                                  
                        test_npix += np.prod(data_clean.shape)
                        self.idcs.append(ct_id)
            else:
                subset = dir_.split('d')[1]
                for ct_id in os.listdir(os.path.join(cur_dir, dir_)):
                    if ct_id.split('_')[-1] == 'clean.npy':
                        ct_id = ct_id.split('_clean.npy')[0]
                        data_clean = np.load(cur_dir+'/fold'+str(subset)+'/'+str(ct_id)+'_clean.npy')
                        self.train_data.append(data_clean)
                        data_labels = np.load(cur_dir+'/fold'+str(subset)+'/'+str(ct_id)+'_label.npy')
                        self.train_labels.append(int(data_labels[-1]))
                        train_pixvlu += np.sum(data_clean)
                        train_pixvar += np.prod(data_clean*data_clean)                                  
                        train_npix += np.prod(data_clean.shape)
        train_pixmean = train_pixvlu / float(train_npix)
        train_pixstd = np.sqrt(train_pixvlu / float(train_npix))
        test_pixmean = test_pixvlu / float(test_npix)
        test_pixstd = np.sqrt(test_pixvlu / float(test_npix))
        self.train_data = np.array(self.train_data)
        self.test_data = np.array(self.test_data)
        self.train_labels = np.array(self.train_labels)
        self.test_labels = np.array(self.test_labels)
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.transform_train = transforms.Compose([
            # transforms.RandomScale(range(28, 38)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomYFlip(),
            transforms.RandomZFlip(),
            transforms.ZeroOut(4),
            transforms.ToTensor(),
            transforms.Normalize((train_pixmean), (train_pixstd)),  # need to cal mean and std, revise norm func
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((test_pixmean), (test_pixstd)),
        ])
        self.transform_train_end = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((train_pixmean), (train_pixstd)),
        ])


    def __getitem__(self, index):
        if self.phase == 'train':
            img, target = self.train_data[index], self.train_labels[index]
            img1 = self.transform_train(img)
            img2 = self.transform_train(img)
            return img1, img2, target
        else:
            img, target = self.test_data[index], self.test_labels[index]
            
            # img, target, idx = self.test_data[index], self.test_labels[index], self.idcs[index]
            img1 = self.transform_test(img)
            img2 = self.transform_test(img)
            this_idx = self.idcs[index]
            return img1, img2, target


    def __len__(self):
        if self.phase=='train':
            return self.train_len
        else:
            return self.test_len
