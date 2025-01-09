import os
import time
import logging
import argparse
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms.functional as T
from utils.path_utils import Path
from glob import glob
from utils.metrics import Evaluator
from PIL import Image
import scipy.io as scio


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_parser():
    parser = argparse.ArgumentParser(description="Gaofen Challenge Training")
    '''
        Model
    '''
    parser.add_argument('--backbone', type=str, default='vgg16',
                        choices=['resnet18', 'resnet50', 'hrnet18', 'vgg16', 'swint', 'mobilenetv2'],
                        help='backbone name')
    '''
        Dataset
    '''
    parser.add_argument('--scales', nargs='+', type=float)
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset name')
    parser.add_argument('--suffix', type=str, default='png',
                        help='dataset suffix')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--groups', type=int, default=120,
                        help='number of regions')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=256,
                        metavar='N', help='input image size for \
                                testing (default: auto)')
    parser.add_argument('--ra_head_num', type=int, default=None,
                        help='number of regions')
    parser.add_argument('--ga_head_num', type=int, default=None,
                        help='number of regions')
    '''
        Checkpoints
    '''
    parser.add_argument('--model_path', type=str, default='/model_best.pth.tar',
                        help='put the path to resuming file if needed')

    parser.add_argument('--save_folder', type=str, default='/results',
                        help='put the path to save prediction')
    args = parser.parse_args()
    return args


def get_logger(save_path):
    logger_name = "main-logger"  # 给日志器取了一个名称
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO，这意味着 logger 会记录 INFO 级别及以上的日志消息，低于 INFO 级别的消息将不会被记录

    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))  # 创建一个文件处理器，将日志写入到指定路径下的 log.txt 文件中。
    log_format = '%(asctime)s %(message)s'  # 设置日志消息的格式，包括时间戳和消息内容。
    fh.setFormatter(logging.Formatter(log_format))  # 将格式应用到文件处理器，确保日志按照指定格式写入文件。
    logger.addHandler(fh)  # 将文件处理器添加到日志器中

    handler = logging.StreamHandler()  # 创建一个控制台处理器，将日志消息输出到控制台。
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"  # 设置在控制台上显示的日志格式，包括时间、日志级别、文件名、行号和消息内容。
    handler.setFormatter(logging.Formatter(fmt))  # 将格式应用到控制台处理器，确保日志在控制台上按照指定格式显示。
    logger.addHandler(handler)  # 将控制台处理器添加到日志器中，以便日志消息被输出到控制台。
    return logger  # 返回配置好的 logger 对象


def main():
    global args, logger
    args = get_parser()

    gray_folder = os.path.join(args.save_folder, 'predict')

    if not os.path.exists(gray_folder):
        os.makedirs(gray_folder)

    logger = get_logger(
        gray_folder)  # 调用之前定义的 get_logger 函数，传递 gray_folder 作为日志文件的保存路径。这将返回一个已经配置好的 logger 对象，可以用于记录日志。
    logger.info(args)  # 获得的 logger 对象上调用 info 方法，记录一条 INFO 级别的日志消息。
    logger.info("=> creating model ...")

    if 'WHUHi' in args.dataset:
        from dataloaders.datasets import WHU_Hi

        IMG_SUFFIX = 'png'

        strlist = str(args.dataset).split('_')

        glob_path = os.path.join(
            '/home1/ywl/PycharmProject/DCN-T/Dataset/WHU-HI/' + strlist[1] + '_' + strlist[2] + '/',
            '*.%s' % (IMG_SUFFIX))

        # glob_path = '../../Dataset/whu_hi/Matlab_data_format/Matlab_data_format/WHU-Hi-' + strlist[
        #     1] + '/WHU_Hi_' + strlist[1] + '.mat'
        test_files = glob(glob_path)  # glob 模块允许你使用类似正则表达式的规则来匹配文件系统中的文件路径。

        in_channels = 3

        if 'LongKou' in strlist[1]:
            prefix = 'LKt'
        elif 'HanChuan' in strlist[1]:
            prefix = 'T'
        elif 'HongHu' in strlist[1]:
            prefix = 'HHCYt'
        else:
            raise NotImplementedError

        target = scio.loadmat(
            '/home1/ywl/PycharmProject/Data_of_HIS_LIDAR/data/classification/Matlab_data_format/Matlab_data_format/WHU-Hi-' +
            strlist[
                1] + '/Training samples and test samples/Test' + strlist[3] + '.mat')[prefix + 'est' + strlist[3]]
        test_data = WHU_Hi.TesDataset(args, target, test_files)
    else:
        raise NotImplementedError

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers,
                                              pin_memory=True)

    if 'LongKou' in args.dataset:
        classes = 9
    elif 'HanChuan' in args.dataset:
        classes = 16
    elif 'HongHu' in args.dataset:
        classes = 22
    else:
        raise NotImplementedError

    from models.network_local_global import rat_model
    model = rat_model(args, classes, in_channels)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True  # 当你设置 cudnn.benchmark = True 时，PyTorch 将尝试寻找在当前配置下最快的算法以优化运行速度。具体来说，它会在每个可用的算法上做一些基准测试，并选择最适合当前输入大小的算法。
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    tes_time1 = time.time()

    test(args, test_loader, test_data.ids, model, classes, test_data.mean, test_data.std, args.crop_size,
         args.crop_size, args.scales, gray_folder)

    tes_time2 = time.time()

    logger.info('[Test Time: %.4f]' % (tes_time2 - tes_time1))


def net_process(model, image, mean, std=None, flip=True):
    if flip:
        input = torch.cat([image, image.flip(3)], 0)  # 2, c, h, w
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]

    output = output.permute(1, 2, 0)

    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2 / 3):
    _, _, ori_h, ori_w = image.shape

    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        # print(mean)

        y = torch.zeros([image.shape[0], 3, crop_h, crop_w]).cuda()
        for k in range(3):
            y[:, [k], :, :] = F.pad(image[:, [k], :, :],
                                    (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half), mode='constant',
                                    value=mean[k])

        image = y

    _, _, new_h, new_w = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)

    prediction_crop = torch.zeros([new_h, new_w, classes], dtype=torch.float).cuda()
    count_crop = torch.zeros([new_h, new_w], dtype=torch.float).cuda()

    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w].clone()

            count_crop[s_h:e_h, s_w:e_w] += 1
            output = net_process(model, image_crop, mean, std)
            prediction_crop[s_h:e_h, s_w:e_w, :] += output

    prediction_crop /= torch.unsqueeze(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w].permute(2, 0,
                                                                                                            1).unsqueeze(
        0)
    prediction = F.interpolate(prediction_crop, (h, w), mode='bilinear')
    prediction = prediction.squeeze(0).permute(1, 2, 0)
    return prediction


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def piece_acc(evaluator, target, pred):
    evaluator.add_batch(target, pred)
    OA = evaluator.Pixel_Accuracy()
    mAcc, Acc = evaluator.Pixel_Accuracy_Class()
    Kappa = evaluator.Kappa()

    return Acc, OA, mAcc, Kappa


from train_memory import all_loss


def test(args, test_loader, data_list, model, classes, mean, std, crop_h, crop_w, scales, gray_folder):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    # data_time = AverageMeter()
    # batch_time = AverageMeter()
    model.eval()
    end = time.time()

    evaluator = Evaluator(classes)
    evaluator.reset()

    if 'WHUHi' in args.dataset:
        vote_all = []
        vote_prob = 0

        p_acc = np.zeros([len(test_loader), classes + 3])

    for i, input in enumerate(test_loader):
        image = input['image']
        target = input['label']
        image = image.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(image)
            print("output.shape", output.shape)
            print("target.shape", target.shape)
            loss = all_loss()(output, target)
            loss = loss.data.clone()
        vote_prob += output.cpu().numpy()
        print("第{}张三光谱的loss为：".format(loss))
    vote_prob = np.argmax(vote_prob, axis=1)
    target = target.cpu().numpy()
    evaluator.add_batch(target, vote_prob)
    OA = evaluator.Pixel_Accuracy()
    print("Test_OA", OA)
    #     if 'WHUHi' in args.dataset:
    #         evaluator_piece = Evaluator(classes)
    #         evaluator_piece.reset()
    #
    #     data_time.update(time.time() - end)
    #     image = input['image']
    #     image = image.cuda()  # B,C,H,W
    #     _, _, h, w = image.shape
    #
    #     prediction = torch.zeros([h, w, classes], dtype=torch.float).cuda()
    #
    #     for scale in scales:
    #         if 'WHUHi' in args.dataset:
    #             # scale = 1
    #             long_size = round(scale * max(h, w))
    #             crop_h = h
    #             crop_w = w
    #         else:
    #             long_size = round(scale * max(h, w))
    #         new_h = long_size
    #         new_w = long_size
    #         if h > w:
    #             new_w = round(long_size / float(h) * w)
    #         else:
    #             new_h = round(long_size / float(w) * h)
    #
    #         image_scale = F.interpolate(image, size=(new_h, new_w), mode='bilinear')
    #
    #         output = scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    #         prediction += output
    #
    #     prediction /= len(scales)
    #     if 'WHUHi' in args.dataset:
    #         vote_prob += prediction
    #
    #     prediction = torch.max(prediction, dim=2)[1].cpu().numpy()
    #
    #     batch_time.update(time.time() - end)
    #     end = time.time()
    #     if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
    #         logger.info('Test: [{}/{}] '
    #                     'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
    #                     'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
    #                                                                                 data_time=data_time,
    #                                                                                 batch_time=batch_time))
    #
    #     target = input['label']
    #     pred = prediction[None, :, :]
    #     target = target.cpu().numpy()  # batch_size * 256 * 256
    #     evaluator.add_batch(target, pred)
    #
    #     if 'WHUHi' in args.dataset:
    #         p_cls_acc, p_oa, p_aa, p_kappa = piece_acc(evaluator_piece, target, pred)
    #         p_acc[i, :classes], p_acc[i, classes], p_acc[i, classes + 1], p_acc[
    #             i, classes + 2] = p_cls_acc, p_oa, p_aa, p_kappa
    #
    #     gray = np.uint8(prediction)
    #     image_path = data_list[i]
    #     image_name = image_path.split('/')[-1].split('.')[0]
    #     gray_path = os.path.join(gray_folder, str(i + 1) + '_' + image_name + '_label.png')
    #     cv2.imwrite(gray_path, gray)
    #
    #     if 'WHUHi' in args.dataset:
    #         vote_all.append(gray)
    #
    # OA = evaluator.Pixel_Accuracy()
    # mIOU, IOU = evaluator.Mean_Intersection_over_Union()
    # mAcc, Acc = evaluator.Pixel_Accuracy_Class()
    # Kappa = evaluator.Kappa()
    #
    # logger.info('************** Accuracy of Images ********************8')
    #
    # logger.info('[Test: OA: %.4f, AA: %.4f, Kappa: %.4f, mIOU: %.4f]' % (
    #     OA, mAcc, Kappa, mIOU))
    #
    # for i in range(classes):
    #     logger.info('Class {}: {:.4f}'.format(i, Acc[i]))
    #
    # ## hard vote
    #
    # if 'WHUHi' in args.dataset:
    #
    #     logger.info('************** Accuracy of Hard Voting ********************')
    #
    #     vote_all = np.stack(vote_all, axis=0)
    #
    #     vote_result = np.zeros([1, vote_all.shape[1], vote_all.shape[2]], dtype=int)
    #
    #     for i in range(vote_all.shape[1]):
    #         for j in range(vote_all.shape[2]):
    #             vote_result[0, i, j] = np.argmax(np.bincount(vote_all[:, i, j]))
    #
    #     evaluator = Evaluator(classes)
    #     evaluator.reset()
    #
    #     evaluator.add_batch(target, vote_result)
    #
    #     OA = evaluator.Pixel_Accuracy()
    #     mIOU, IOU = evaluator.Mean_Intersection_over_Union()
    #     mAcc, Acc = evaluator.Pixel_Accuracy_Class()
    #     Kappa = evaluator.Kappa()
    #
    #     logger.info('[Test: OA: %.4f, AA: %.4f, Kappa: %.4f, mIOU: %.4f]' % (
    #         OA, mAcc, Kappa, mIOU))
    #
    #     for i in range(classes):
    #         logger.info('Class {}: {:.4f}'.format(i, Acc[i]))
    #
    #     vote = np.uint8(vote_result[0])
    #     vote_path = os.path.join(gray_folder, args.dataset + '_hard_vote.png')
    #     # cv2.imwrite(vote_path, vote)
    #
    #     logger.info('************** Accuracy of Soft Voting ********************')
    #
    #     vote_prediction = torch.max(vote_prob, dim=2)[1].cpu().numpy()
    #     vote_pred = vote_prediction[None, :, :]
    #
    #     evaluator = Evaluator(classes)
    #     evaluator.reset()
    #     evaluator.add_batch(target, vote_pred)
    #
    #     OA = evaluator.Pixel_Accuracy()
    #     mIOU, IOU = evaluator.Mean_Intersection_over_Union()
    #     mAcc, Acc = evaluator.Pixel_Accuracy_Class()
    #     Kappa = evaluator.Kappa()
    #
    #     logger.info('[Test: OA: %.4f, AA: %.4f, Kappa: %.4f, mIOU: %.4f]' % (
    #         OA, mAcc, Kappa, mIOU))
    #
    #     for i in range(classes):
    #         logger.info('Class {}: {:.4f}'.format(i, Acc[i]))
    #
    #     vote = np.uint8(vote_pred[0])
    #     vote_path = os.path.join(gray_folder, args.dataset + '_soft_vote.png')
    #     # cv2.imwrite(vote_path, vote)
    #
    #     ## save piece acc
    #
    #     logger.info('**** Save piece accuracies! ****')
    #
    #     scio.savemat(os.path.join(gray_folder, args.dataset + '_piece_acc.mat'), {'data': p_acc})
    #
    # logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
