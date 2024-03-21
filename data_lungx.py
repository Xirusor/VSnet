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
import cv2

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
        self.fold = fold
        self.phase = phase
        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []
        self.idcs = []

        train_pixvlu,train_npix,train_pixvar = 0,0,0
        test_pixvlu,test_npix,test_pixvar = 0,0,0
        cur_dir = '/home/zhangconghao/VUnet_1/data/LUNGx/data'
        filename = os.listdir(cur_dir+'/images')
        filename.sort(key=lambda x:x[:-4])

        for dir_ in filename:
            # data_clean = np.repeat(cv2.imread(os.path.join(cur_dir, dir_))[15:47,15:47,0:2],16, axis=2)
            data_clean = np.load(os.path.join(cur_dir+'/images', dir_))

            self.test_data.append(data_clean)
            test_pixvlu += np.sum(data_clean)
            test_pixvar += np.prod(data_clean*data_clean)                                  
            test_npix += np.prod(data_clean.shape)
            # self.idcs.append(ct_id)
            
        labellist = os.listdir(cur_dir+'/labels')
        labellist.sort(key=lambda x:x[:-4])
        for dir_ in labellist:
            data_labels = np.load(os.path.join(cur_dir+'/labels', dir_))
            self.test_labels.append(int(data_labels))
        
        test_pixmean = test_pixvlu / float(test_npix)
        test_pixstd = np.sqrt(test_pixvlu / float(test_npix))
        self.test_data = np.array(self.test_data)

        self.test_labels = np.array(self.test_labels)

        self.test_len = len(self.test_data)

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((test_pixmean), (test_pixstd)),
        ])

    def __getitem__(self, index):
            # img, target = self.test_data[index], self.test_labels[index]
            img, target = self.test_data[index], self.test_labels[index]
            img = self.transform_test(img)
            # img, target, idx = self.test_data[index], self.test_labels[index], self.idcs[index]
            # this_idx = self.idcs[index]
            return img, target


    def __len__(self):
        return self.test_len
