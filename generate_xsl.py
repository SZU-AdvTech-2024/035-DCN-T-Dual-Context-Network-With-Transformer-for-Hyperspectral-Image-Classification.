import xlwt
# from scipy.io import loadmat
import os
import argparse
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score
import hdf5storage as hdf5

RUN = 1


# 计算模型的recall, AA, OA and kappa
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统计数据')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    parser.add_argument('--spc', type=str, default=20,
                        help='SAMPLE SIZE')
    parser.add_argument('--patch', type=str, default=11,
                        help='PATCH SIZE')
    parser.add_argument('--range_vote', type=str, default=11,
                        help='RANGE VOTE')
    arg = parser.parse_args()
    dataset_name = arg.name
    # 创建工作簿
    book = xlwt.Workbook()
    # 创建sheet
    sheet = book.add_sheet('metric')
    # gt_root = "trainTestSplit/{}".format(dataset_name)
    for r in range(RUN):
        # 读取真实gt
        # gt_path = os.path.join(gt_root, 'sample{}_run{}.mat'.format(arg.spc, r))
        # m = hdf5.loadmat(gt_path)
        # gt = m['test_gt']
        te_path = '/home3/ywl/PycharmProject/CVSSN/data/YC_Compete/test_label.mat'
        te = hdf5.loadmat(te_path)
        gt = te['test_label']
        # 读取预测标签
        # m = loadmat('{}_{}_{}_{}.mat'.format(SAMPLES, r, HIDDEN_SIZE, PATCH_SIZE))
        m = hdf5.loadmat('/home1/ywl/PycharmProject/DCN-T/prediction/8456/0.mat')
        pred = m['pred']
        # if pred.min() == 0:
        #     pred += 1
        # 下标
        indices = np.nonzero(gt)
        ans = measure(pred[indices], gt[indices])
        recall = ans['class_recall']
        AA = ans['AA']
        OA = ans['OA']
        kappa = ans['kappa']
        i = 0
        while i < len(recall):
            sheet.write(r, i, recall[i])
            i += 1
        sheet.write(r, i, OA)
        i += 1
        sheet.write(r, i, AA)
        i += 1
        sheet.write(r, i, kappa)
    book.save('xsl/ps_{}_8456.xls'.format(arg.patch))
    print('*' * 5 + 'FINISH' + '*' * 5)

'''
python generate_xsl.py --name YC_Compete --patch 11 --range_vote 5
python generate_xsl.py --name YC_Compete --patch 7 --range_vote 7

'''
