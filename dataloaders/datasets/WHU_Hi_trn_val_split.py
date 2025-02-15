import os
import torch
import scipy.io as scio
import numpy as np
from PIL import Image
from torch.utils import data
from utils.path_utils import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from glob import glob

class TrainDataset(data.Dataset):
    def __init__(self, args, target, split_files=None):
        self.args=args
        self.ids=split_files
        self.target = target
        self.mean=(0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)
        print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
             #tr.RandomCrop(self.args.crop_size),
             #tr.RandomHorizontalFlip(),
             tr.Normalize(mean=self.mean, std=self.std),
             tr.ToTensor()])

    def _class_to_trainid(self, label):

        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        img_path = self.ids[i]

        _img = Image.open(img_path)

        #_img = np.load(img_path)

        _target = self.target

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample= self.transforms(sample)

        return sample


class ValDataset(data.Dataset):

    def __init__(self, args, target, split_files=None):
        self.args=args
        self.ids=split_files
        self.target = target
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
            #tr.CenterCrop(self.args.crop_size),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

    def _class_to_trainid(self, label):

        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        img_path = self.ids[i]

        _img = Image.open(img_path)

        #_img = np.load(img_path)

        _target = self.target

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample = self.transforms(sample)
        return sample

class TesDataset(data.Dataset):

    def __init__(self, args, target,split_files=None):
        self.args=args
        self.ids=split_files
        self.target = target
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        print('Creating dataset with {} examples'.format(len(self.ids)))
        self.transforms = transforms.Compose([
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

    def _class_to_trainid(self, label):
        label_copy = label.copy()

        label_copy -= 1

        label_copy[label_copy < 0] = 255

        return label_copy

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_path = self.ids[i]

        _img = Image.open(img_path)

        #_img = np.load(img_path)

        _target = self.target

        _target = self._class_to_trainid(_target)

        _target = Image.fromarray(_target)

        sample = {'image': _img, 'label': _target}

        sample = self.transforms(sample)

        return sample


def production_label(num, y_map, w, split='Trn_Val'):

	num = np.array(num)
	idx_2d = np.zeros([num.shape[0], 2]).astype(int)
	idx_2d[:, 0] = num // w
	idx_2d[:, 1] = num % w

	label_map = np.zeros(y_map.shape)
	for i in range(num.shape[0]):
		label_map[idx_2d[i,0],idx_2d[i,1]] = y_map[idx_2d[i,0],idx_2d[i,1]]

	print('{} label map preparation Finished!'.format(split))
	
	return label_map 
	        
def trn_val_split(y_trn_data):

    h, w = y_trn_data.shape

    all_trn_num = np.where(y_trn_data.reshape(-1) > 0)[0]
    
    np.random.shuffle(all_trn_num)
    
    ix = int(all_trn_num.shape[0] * 0.8)
    
    trn_num = all_trn_num[:ix]
    
    val_num = all_trn_num[ix:]
    
    y_trn_map = production_label(trn_num, y_trn_data, w, split='Trn')
    
    y_val_map = production_label(val_num, y_trn_data, w, split='Val')
    
    print('Number of Training labels:{}'.format(trn_num.shape[0]))
    
    print('Number of Validation labels:{}'.format(val_num.shape[0]))
    
    return y_trn_map, y_val_map

def make_data_loader(args):
    IMG_SUFFIX = 'png'

    strlist = str(args.dataset).split('_')

    glob_path = os.path.join('../../Dataset/whu_hi/whuhi_image_2percent/' + strlist[1] + '_' + strlist[2] + '/', '*.%s' % (IMG_SUFFIX))
    
    print(glob_path)

    trn_file = glob(glob_path)  # extract all the .png to a list

    if 'LongKou' in strlist[1]:
        prefix = 'LKt'
    elif 'HanChuan' in strlist[1]:
        prefix = 'T'
    elif 'HongHu' in strlist[1]:
        prefix = 'HHCYt'
    else:
        raise NotImplementedError

    trn_target = scio.loadmat('../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-'+strlist[1]+'/Training samples and test samples/Train'+strlist[3]+'.mat')[prefix+'rain'+strlist[3]]
    
    #trn_files = trn_file + trn_file
    
    
    #trn_target, val_target = trn_val_split(target)

    ix = int(len(trn_file) * 0.95)
    
    train_set = TrainDataset(args, trn_target, trn_file)

    val_target = scio.loadmat(
        '../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-' + strlist[
            1] + '/Training samples and test samples/Test' + strlist[3] + '.mat')[prefix + 'est' + strlist[3]]

    val_set = ValDataset(args, val_target, trn_file)
    
    trn_num = np.where(trn_target.reshape(-1) > 0)[0]
    
    tes_num = np.where(val_target.reshape(-1) > 0)[0]
    
    print('Number of Training labels:{}'.format(trn_num.shape[0]))
    
    print('Number of Validation labels:{}'.format(tes_num.shape[0]))
    

    if args.distributed=='True':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                      num_replicas=args.world_size,rank=args.rank)#分布式采样器
    else:
        train_sampler = None

    if args.distributed=='True':
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set,
                                                                      num_replicas=args.world_size,
                                                                      rank=args.rank)  # 分布式采样器
    else:
        val_sampler = None

    if args.distributed == 'True':
        args.batch_size = int(args.batch_size / args.world_size)#将一个节点的BS按GPU平分
        args.test_batch_size = int(args.test_batch_size / args.world_size)
        args.workers = int(args.workers / args.world_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, sampler=val_sampler, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader

