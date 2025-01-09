import os
import argparse
import time
# import apex
import logging
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.multiprocessing
import torch.distributed as dist
from models.sync_batchnorm.replicate import patch_replication_callback
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
import scipy.io as scio
from glob import glob
import visdom
import hdf5storage as hdf5


# torch.cuda.set_device(1)

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 设定程序中可见的 GPU 设备为第二个 GPU


# def get_logger(save_path):
#     logger_name = "main-logger"
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.INFO)
#     fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
#     log_format = '%(asctime)s %(message)s'
#     fh.setFormatter(logging.Formatter(log_format))
#     logger.addHandler(fh)
#
#     handler = logging.StreamHandler()
#     fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
#     handler.setFormatter(logging.Formatter(fmt))
#     logger.addHandler(handler)
#     return logger

def main_process(args):
    return not args.distributed == 'True' or (args.distributed == 'True' and args.rank % args.world_size == 0)


class all_loss(nn.Module):
    def __init__(self):
        super().__init__()

        # ce
        self.criterion_1 = nn.CrossEntropyLoss(
            ignore_index=255)  # ignore_index 是 nn.CrossEntropyLoss 的一个参数，用于指定忽略某些特定标签的损失计算
        self.criterion_2 = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, output, target):

        if len(output) == 2:
            output_1, output_2 = output

            target = target.cuda(
                non_blocking=True).long()  # non_blocking=True 表示异步执行，使得数据传输过程中的 CPU 等待时间减少，而无需等待 GPU 数据传输完成。

            loss1 = self.criterion_1(output_1, target)
            loss2 = self.criterion_2(output_2, target)

            #            loss = loss1 + 0.4*loss2

            return loss1, loss2

        else:

            output_1 = output

            target = target.cuda(non_blocking=True).long()

            loss1 = self.criterion_1(output_1, target)

            loss = loss1

            return loss


class Trainer(object):
    def __init__(self, args, LOCAL_RANK=0, r=0):
        self.args = args

        # Define Saver
        # if main_process(args):
        #     self.saver = Saver(args)
        #     self.saver.save_experiment_config()
        #     self.logger = get_logger(self.saver.experiment_dir)

        # print args
        # if main_process(args):
        #     self.logger.info(args)

        # Define Dataloader
        if 'WHUHi' in self.args.dataset:
            from dataloaders.datasets.WHU_Hi import make_data_loader
            self.train_loader, self.val_loader = make_data_loader(args, r)
        else:
            raise NotImplementedError

        in_channels = 3

        if 'LongKou' in self.args.dataset:
            classes = 9
        elif 'HanChuan' in self.args.dataset:
            classes = 16
        elif 'HongHu' in self.args.dataset:
            classes = 22
        elif 'jiaqiyue' in self.args.dataset:
            classes = 11
        elif 'jiaqicang' in self.args.dataset:
            classes = 14
        elif 'YCcompete' in self.args.dataset:
            classes = 18
        else:
            raise NotImplementedError

        # Define model

        from models.network_local_global import rat_model

        model = rat_model(args, classes, in_channels)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        model.cuda()

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)  # nesterov 加快收敛速度，减少震荡

        # optimizer = torch.optim.Adam(train_params,weight_decay=args.weight_decay)

        # Define Criterion

        self.criterion = all_loss()

        # Define Evaluator

        self.evaluator = Evaluator(classes)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args, args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Define accuracy

        if 'WHUHi' in self.args.dataset:
            self.val_vote_acc = []
            self.evaluator_vote = Evaluator(classes)
        else:
            self.val_vote_acc = None

        # Define train form

        if args.distributed == 'True':
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[LOCAL_RANK],
                                                              find_unused_parameters=True)
            # if main_process(args):
            #     self.logger.info("Implementing distributed training!")
        else:
            # if args.use_apex == 'True':  # nvidia 的 apex
            #     model = apex.parallel.convert_syncbn_model(model)
            #     model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.opt_level)
            #     if main_process(args):
            #         self.logger.info("Implementing parallel hybrid training!")
            model = torch.nn.DataParallel(model.cuda())  # 普通的单机多卡, device_ids=[1]
            patch_replication_callback(model)
            # if main_process(args):
            #     self.logger.info("Implementing parallel training!")

        self.model, self.optimizer = model, optimizer

        # Resuming checkpoint
        self.best_pred = 0.0

        if args.ft == 'True':  # finetuning on a different dataset
            if not os.path.isfile(args.resume):
                if main_process(args):
                    raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            else:
                # if main_process(args):
                # self.logger.info("=> loading ft model...")

                checkpoint = torch.load(args.resume, map_location='cpu')
                ckpt_dict = checkpoint['state_dict']
                model_dict = {}

                state_dict = model.state_dict()
                for k, v in ckpt_dict.items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)

                if args.distributed == 'True':
                    self.model.load_state_dict(state_dict)
                else:
                    self.model.module.load_state_dict(state_dict)

                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # self.best_pred = checkpoint['best_pred']
                if main_process(args):
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft == 'True':
            self.args.start_epoch = 0

        self.best_test_OA = 0.0  # 用于存储最佳 mIOU

        self.model_dir = f'{args.dataset}'

        # 检查文件夹是否存在，如果不存在则创建文件夹
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 将 self.best_model_path 拼接到 self.model_dir 目录下
        self.best_model_path = os.path.join(self.model_dir, f"best_20_{r}_MS.pkl")
        self.r = r

    def training(self, epoch, args, trn_loss):
        torch.cuda.empty_cache()
        epoch_loss = 0.0
        start_time = time.time()
        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            image = sample['image']
            target = sample['label']#.transpose(2, 1)
            image = image.cuda(
                non_blocking=True)  # non_blocking参数来控制数据传输是否异步进行。如果non_blocking参数为True，并且源数据在锁页内存中，那么数据传输将与主机异步进行，即不会阻塞主机的其他操作。如果non_blocking参数为False，或者源数据不在锁页内存中，那么数据传输将与主机同步进行，即会等待数据传输完成后再执行主机的其他操作。
            self.scheduler(self.optimizer, i, epoch)
            output = self.model(image)
            loss1, loss2 = self.criterion(output, target)

            loss = loss1 + 0.4 * loss2

            reduced_loss = loss.data.clone()

            trn_loss.append(loss1.item())

            if self.args.distributed == 'True':
                reduced_loss = reduced_loss / args.world_size
                dist.all_reduce_multigpu([reduced_loss])

            self.optimizer.zero_grad()

            epoch_loss += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=20, norm_type=2)  # 梯度裁剪

            self.optimizer.step()

            if main_process(self.args):
                tbar.set_description('Training batch: %d' % (i + 1))
                tbar.set_postfix(Loss=epoch_loss / (i + 1))

        end_time = time.time()

        torch.cuda.empty_cache()

        print('Training epoch [{}/{}]: Loss: {:.4f}. Cost {:.4f} secs'.format(epoch + 1, self.args.epochs,
                                                                              epoch_loss * 1.0 / (i + 1),
                                                                              end_time - start_time))

        return trn_loss

    def validation(self, epoch, val_loss, isbest=True):
        if main_process(self.args):
            print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        torch.cuda.empty_cache()
        self.model.eval()
        self.evaluator.reset()

        if 'WHUHi' in self.args.dataset:
            vote_prob = 0

        if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
            preds = []

        tbar = tqdm(self.val_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image = sample['image']
            target = sample['label']
            image = image.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(image)
                loss = self.criterion(output, target)
                val_loss.append(loss.item())

            if main_process(self.args):
                tbar.set_description('Validation batch: %d' % (epoch))

            if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
                preds.append(output.cpu().numpy().argmax(axis=1))  # b, h, w

            if 'WHUHi' in self.args.dataset and self.args.mode == 'soft':
                vote_prob += output.cpu().numpy()  # 1,c,h,w

        if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
            preds = np.concatenate(preds, axis=0).astype('int')  # B, h, w
            _, h, w = preds.shape
            vote_pred = np.zeros([1, h, w]).astype('int')

            for ii in range(h):
                for jj in range(w):
                    vote_pred[0, ii, jj] = np.argmax(np.bincount(preds[:, ii, jj]))

        if 'WHUHi' in self.args.dataset and self.args.mode == 'soft':  # soft voting
            vote_pred = np.argmax(vote_prob, axis=1)  # 1,h,w

        target = target.cpu().numpy()  # batch_size * 256 * 256
        print("target.shape", target.shape)
        self.evaluator.add_batch(target, vote_pred)

        if 'WHUHi' in self.args.dataset:
            OA = self.evaluator.Pixel_Accuracy()
            mIOU, IOU = self.evaluator.Mean_Intersection_over_Union()
            mAcc, Acc = self.evaluator.Pixel_Accuracy_Class()
            Kappa = self.evaluator.Kappa()
            self.val_vote_acc.append(OA)
            if main_process(self.args):
                print('[Val Vote: OA: %.4f]' % (OA))

        if main_process(self.args):
            print('[Epoch: %d, Val OA: %.4f, mIOU: %.4f, mAcc: %.4f, Kappa: %.4f]' % (
                epoch, OA, mIOU, mAcc, Kappa))
        torch.cuda.empty_cache()
        return val_loss, OA

    def test(self, epoch):
        torch.cuda.empty_cache()
        from dataloaders.datasets import WHU_Hi

        IMG_SUFFIX = 'png'

        strlist = str(self.args.dataset).split('_')

        glob_path = os.path.join(
            '/home2/lzn/project/DCN-T/data/Matlab_data_format/save/' + strlist[1] + '_' +
            strlist[
                2] + '/',
            '*.%s' % (IMG_SUFFIX))

        test_files = glob(glob_path)  # glob 模块允许你使用类似正则表达式的规则来匹配文件系统中的文件路径。

        if 'LongKou' in strlist[1]:
            prefix = 'LKt'
        elif 'HanChuan' in strlist[1]:
            prefix = 'T'
        elif 'HongHu' in strlist[1]:
            prefix = 'HHCYt'
        # else:
        #     raise NotImplementedError
        if "jiaqiyue" in self.args.dataset:
            target = \
                scio.loadmat(f'/home5/ywl/PycharmProject/MS_comparison/trainTestSplit/Yue/sample20_run{self.r}.mat')[
                    'test_gt']
        elif "jiaqicang" in self.args.dataset:
            target = \
                scio.loadmat(f'/home5/ywl/PycharmProject/MS_comparison/trainTestSplit/Cang/sample20_run{self.r}.mat')[
                    'test_gt']
        elif "YCcompete" in self.args.dataset:
            target = hdf5.loadmat('/home3/ywl/PycharmProject/CVSSN/data/YC_Compete/test_label.mat')['test_label']
        else:
            target = scio.loadmat(
                '/home2/lzn/project/DCN-T/data/Matlab_data_format/WHU-Hi-' +
                strlist[
                    1] + '/Training samples and test samples/Test' + strlist[3] + '.mat')[prefix + 'est' + strlist[3]]
        test_data = WHU_Hi.TesDataset(self.args, target, test_files)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=self.args.workers,
                                                  pin_memory=True)

        if main_process(self.args):
            print('>>>>>>>>>>>>>>>> Start Test >>>>>>>>>>>>>>>>')
        self.model.eval()
        self.evaluator.reset()

        if 'WHUHi' in self.args.dataset:
            vote_prob = 0

        if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
            preds = []

        tbar = tqdm(test_loader, desc='\r')
        all_loss = 0
        for i, sample in enumerate(tbar):
            image = sample['image']
            target = sample['label']#.transpose(2, 1)
            image = image.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(image)
                loss = self.criterion(output, target)
                all_loss += loss

            if main_process(self.args):
                tbar.set_description('Test batch: %d' % (epoch))

            if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
                preds.append(output.cpu().numpy().argmax(axis=1))  # b, h, w

            if 'WHUHi' in self.args.dataset and self.args.mode == 'soft':
                vote_prob += output.cpu().numpy()  # 1,c,h,w

        if 'WHUHi' in self.args.dataset and self.args.mode == 'hard':
            preds = np.concatenate(preds, axis=0).astype('int')  # B, h, w
            _, h, w = preds.shape
            vote_pred = np.zeros([1, h, w]).astype('int')

            for ii in range(h):
                for jj in range(w):
                    vote_pred[0, ii, jj] = np.argmax(np.bincount(preds[:, ii, jj]))

        if 'WHUHi' in self.args.dataset and self.args.mode == 'soft':  # soft voting
            vote_pred = np.argmax(vote_prob, axis=1)  # 1,h,w

        target = target.cpu().numpy()  # batch_size * 256 * 256
        # print("target.shape", target.shape)
        self.evaluator.add_batch(target, vote_pred)

        if 'WHUHi' in self.args.dataset:
            OA = self.evaluator.Pixel_Accuracy()
            mIOU, IOU = self.evaluator.Mean_Intersection_over_Union()
            mAcc, Acc = self.evaluator.Pixel_Accuracy_Class()
            Kappa = self.evaluator.Kappa()
            self.val_vote_acc.append(OA)
            if main_process(self.args):
                print('[Test Vote: OA: %.4f]' % (OA))

        if main_process(self.args):
            print('[Epoch: %d, Test OA: %.4f, mIOU: %.4f, mAcc: %.4f, Kappa: %.4f, All_loss:%.4f]' % (
                epoch, OA, mIOU, mAcc, Kappa, all_loss))

        # 检查当前 mIOU 是否是最佳结果，如果是，则保存模型
        if OA > self.best_test_OA:
            print(f"New best OA found: {OA:.4f}, saving model...")
            self.best_test_OA = OA
            torch.save(self.model.state_dict(), self.best_model_path)  # 保存模型权重
        else:
            print(f"Not new best OA found,best OA is: {self.best_test_OA:.4f}")
        torch.cuda.empty_cache()

        return OA


def main():
    parser = argparse.ArgumentParser(description="Gaofen Challenge Training")
    '''
        Model
    '''
    parser.add_argument('--backbone', type=str, default='vgg16',
                        choices=['resnet18', 'resnet50', 'vgg16', 'hrnet18', 'vitaev2_s', 'mobilenetv2', 'swint'],
                        help='backbone name')
    '''
        Dataset
    '''
    ## WHUHi + district + channel + sample
    ## eg: WHUHi_LongKou_10_100

    parser.add_argument('--dataset', type=str, default='WHUHi_LongKou_15_100', help='dataset name')
    parser.add_argument('--workers', type=int, default=2,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='crop image size')

    '''
        Hyper Parameters
    '''
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--groups', type=int, default=128,
                        help='number of regions')
    parser.add_argument('--ra_head_num', type=int, default=4,
                        help='number of regions')
    parser.add_argument('--ga_head_num', type=int, default=4,
                        help='number of regions')
    '''
        Optimizer
    '''
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    '''
        Fine-tune
    '''
    parser.add_argument('--ft', type=str, default='False',
                        choices=['True', 'False'],
                        help='finetuning on a different dataset')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--freeze_bn', action='store_true', default=False,
                        help='whether freeze bn while finetuning')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='whether freeze backbone while finetuning')
    '''
        Evaluation
    '''
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    '''
        Others
    '''
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    '''
        apex
    '''
    parser.add_argument('--use_apex', type=str, default='False',
                        choices=['True', 'False'], help='use apex')
    parser.add_argument('--opt_level', type=str, default='O0',
                        choices=['O0', 'O1', 'O2', 'O3'], help='hybrid training')
    '''
        distributed
    '''
    parser.add_argument('--distributed', type=str, default='False',
                        choices=['True', 'False'], help='distributed training')
    parser.add_argument('--local_rank', type=int,
                        default=0)  # 进程内，GPU 编号，非显式参数，由 torch.distributed.launch 内部指定。比方说， rank = 3，local_rank = 0 表示第 3 个进程内的第 1 块 GPU。

    '''
        mode
    '''
    parser.add_argument('--mode', type=str, default='soft',
                        choices=['soft', 'hard'], help='voting mode')

    args = parser.parse_args()

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    for r in range(1):

        if args.distributed == 'True':
            print('here?')
            exit()
            args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
            args.rank = int(os.environ["RANK"])
            LOCAL_RANK = int(os.environ['LOCAL_RANK'])  # args.rank % torch.cuda.device_count()
            dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size,
                                    rank=args.rank)  # 分布式TCP初始化
            torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
            trainer = Trainer(args, LOCAL_RANK, r)
        else:
            print("12344")
            trainer = Trainer(args)

        trn_loss = []
        val_loss = []
        trn_time = 0

        # viz = visdom.Visdom(env="DCN-T")
        # if not viz.check_connection:
        #     print("visdom is not connected.Did you run 'python -m visdom.server'?")

        win = None
        win1 = None
        win2 = None

        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trn_time1 = time.time()
            trn_loss = trainer.training(epoch, args, trn_loss)
            trn_time2 = time.time()
            # if not trainer.args.no_val and epoch % args.eval_interval == 0:
            #     val_loss, val_acc = trainer.validation(epoch, val_loss)
            test_acc = trainer.test(epoch)
            trn_time = trn_time + trn_time2 - trn_time1

        tes_time1 = time.time()

        tes_time2 = time.time()
        tes_time = tes_time2 - tes_time1

        print('[Trn time: %.4f]' % (trn_time))
        print('[Tes time: %.4f]' % (tes_time))


if __name__ == '__main__':
    main()

# nproc_per_node 参数指定为当前主机创建的进程数。一般设定为当前主机的 GPU 数量
# nnodes 参数指定当前 job 包含多少个节点
# node_rank 指定当前节点的优先级
# master_addr 和 master_port 分别指定 master 节点的 ip:port

'''
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes 1 \
    --node_rank=0 --master_port=2096 --use_env train_memory.py \
    --dataset 'WHUHi_HongHu_15_300' \
    --backbone 'resnet50' \
    --epochs 15 --lr 1e-3 --groups 128 --eval_interval 1 \
    --batch_size 1 --test_batch_size 1 --workers 2 \
    --ra_head_num 4 --ga_head_num 4 --mode 'soft'
    
Cang:

CUDA_VISIBLE_DEVICES=5,4 nohup python -m torch.distributed.launch --nproc_per_node=1 --nnodes 1 \
    --node_rank=0 --master_port=1902 --use_env train_memory.py \
    --distributed True\
    --dataset 'WHUHi_jiaqiyue_15_25' \
    --backbone 'vgg16' \
    --epochs 30 --lr 1e-3 --groups 128 --eval_interval 1 \
    --batch_size 4 --test_batch_size 1 --workers 2 \
    --ra_head_num 4 --ga_head_num 4 --mode 'soft'> yue.txt 2>&1 &
    
'''
