'''Ensemble Prediction'''
import torch
from torch.nn import functional as F
import argparse
from configparser import ConfigParser
from torch.utils.data import DataLoader
import numpy as np
import os
from models.network_local_global import rat_model
import torchvision.transforms as T
from PIL import Image

from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score
import torch.nn as nn
from torch.utils.data import Dataset

from tqdm import tqdm
import hdf5storage as hdf5
import glob


class Classifier(nn.Module):
    def __init__(self, model, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.model = model
        self.fc = nn.Linear(input_dim, num_classes)  # Assuming model output is flattened to input_dim size

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


def get_classification_map(y_pred, y):
    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k] + 1
                k += 1

    return cls_labels


def test(args):
    torch.cuda.empty_cache()
    from dataloaders.datasets import WHU_Hi

    IMG_SUFFIX = 'png'

    strlist = str(args.dataset).split('_')

    glob_path = os.path.join(
        '/home1/ywl/PycharmProject/DCN-T/Dataset/WHU-HI/' + strlist[1] + '_' + strlist[2] + '/',
        '*.%s' % (IMG_SUFFIX))

    test_files = glob.glob(glob_path)  # glob 模块允许你使用类似正则表达式的规则来匹配文件系统中的文件路径。

    target = hdf5.loadmat('/home3/ywl/PycharmProject/CVSSN/data/YC_Compete/test_label.mat')['test_label']

    test_data = WHU_Hi.TesDataset(args, target, test_files)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    return test_loader


def measure(y_pred, y_true):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.cpu().numpy()
    else:
        y_pred = np.array(y_pred)
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.cpu().numpy()
    else:
        y_true = np.array(y_true)
    # 计算类别 recall 值
    class_recall = recall_score(y_true, y_pred, average=None)
    # 计算平均 recall
    AA = class_recall.mean()
    # 计算准确率
    OA = accuracy_score(y_true, y_pred)
    # 计算 kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    res = {'class_recall': class_recall.tolist(),
           'AA': AA,
           'OA': OA,
           'kappa': kappa}
    return res


from collections import OrderedDict


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='Cang',
                        help='DATASET NAME')
    parser.add_argument('--spc', type=int, default=20,
                        help='SAMPLE PER CLASS')
    parser.add_argument('--r', type=int, default=5,
                        help='RUN')
    parser.add_argument('--patch', type=int, default=32,
                        help='PATCH SIZE')
    parser.add_argument('--batch', type=int, default=128,
                        help='BATCH SIZE')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='NUM WORKER')
    parser.add_argument('--seed', type=int, default=666,
                        help='RANDOM SEED')
    arg = parser.parse_args()
    arg.backbone = 'vgg16'
    arg.mode = 'soft'
    arg.ra_head_num = 4
    arg.ga_head_num = 4
    arg.groups = 128
    arg.dataset = 'WHUHi_YCcompete_15'
    print(arg)
    # 设置随机种子
    SEED = 971104
    torch.manual_seed(SEED)

    net_save_root = f'/home1/ywl/PycharmProject/DCN-T/best_model_84.56.pth'
    pred_save_root = f'prediction/8456'
    if not os.path.exists(pred_save_root):
        os.makedirs(pred_save_root)

    net = rat_model(arg, 18, 3)
    net_path = net_save_root
    # 加载原始的 state_dict
    state_dict = torch.load(net_path)
    # 加载到模型中
    state_dict = remove_module_prefix(state_dict)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net.cuda()
    preds = []

    loader = test(arg)
    vote_prob = 0
    with torch.no_grad():
        for sample in tqdm(loader, desc='Processing batches', leave=False):
            image, label = sample['image'].cuda(), sample['label'].cuda()
            outputs = net(image)
            vote_prob += outputs.cpu().numpy()

    map = np.argmax(vote_prob, axis=1).squeeze(0)+1  # 1,h,w
    test_label = hdf5.loadmat('/home3/ywl/PycharmProject/CVSSN/data/YC_Compete/test_label.mat')['test_label']
    # map = get_classification_map(preds, test_label)
    # 计算OA，AA，kappa
    # ================
    indices = np.nonzero(test_label)
    ans = measure(map[indices], test_label[indices])
    recall = ans['class_recall']
    AA = ans['AA']
    OA = ans['OA']
    kappa = ans['kappa']
    print(recall, AA, OA, kappa)
    hdf5.savemat(os.path.join(pred_save_root, '{}.mat'.format(0)),
                 {'pred': map, 'OA:': OA, 'AA:': AA, 'kappa': kappa})

    print('-' * 5 + 'FINISH' + '-' * 5)

"""

CUDA_VISIBLE_DEVICES=6 python pred_YC.py --cfg configs/internimage_g_22kto1k_512.yaml --patch 11

"""
'''

    from collections import OrderedDict
    def remove_module_prefix(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict
        
CUDA_VISIBLE_DEVICES=2 python pred_YC.py

'''
