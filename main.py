import pandas as pd
import numpy as np

import argparse
import torch
from torchvision import transforms as trn
from torch.utils.data import DataLoader

from torch import nn, optim
from tqdm import tqdm
from utils import AverageMeter, gap

import os
import logging
from datetime import datetime
from tensorboardX import SummaryWriter
from autoaugment import ImageNetPolicy

from dataset import LandmarkDataset, TestDataset
from models import Resnet50, Efficientnet
import torch_optimizer as toptim
from pytorch_models.timm.models import gluon_seresnext50_32x4d,gluon_seresnext101_32x4d
from pytorch_models.timm.models.vision_transformer import  vit_base_patch16_224, vit_small_patch16_224
from Ranger.ranger.ranger2020 import Ranger

def init_logger(save_dir, comment=None):
    c_date, c_time = datetime.now().strftime("%Y%m%d/%H%M%S").split('/')
    if comment is not None:
        if os.path.exists(os.path.join(save_dir, c_date, comment)):
            comment += f'_{c_time}'
    else:
        comment = c_time
    log_dir = os.path.join(save_dir, c_date, comment)
    log_txt = os.path.join(log_dir, 'log.txt')

    os.makedirs(f'{log_dir}/ckpts')
    os.makedirs(f'{log_dir}/submissions')

    global writer
    writer = SummaryWriter(log_dir)
    global logger
    logger = logging.getLogger(c_time)

    logger.setLevel(logging.INFO)
    logger = logging.getLogger(c_time)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    h_file = logging.FileHandler(filename=log_txt, mode='a')
    h_file.setFormatter(fmt)
    h_file.setLevel(logging.INFO)
    logger.addHandler(h_file)
    logger.info(f'Log directory ... {log_txt}')
    return log_dir


def train(model, loader, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    pbar = tqdm(loader, ncols=150)
    y_true = dict()
    y_pred = dict()


    for i, (image, iid, target, _) in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs = model(image.cuda())
        loss = criterion(outputs, target.cuda())
        loss.backward()
        optimizer.step()


        conf, indice = torch.topk(outputs, k=5)
        indice = indice.cpu()

        y_true.update({k: t for k, t in zip(iid, target.numpy())})
        y_pred.update({k: (t, c) for k, t, c in
                       zip(iid, indice[:, 0].cpu().detach().numpy(), conf[:, 0].cpu().detach().numpy())})

        top1.update(torch.sum(indice[:, :1] == target.view(-1, 1)).item())
        top5.update(torch.sum(indice == target.view(-1, 1)).item())
        losses.update(loss)

        log = f'[Epoch {epoch}] '
        log += f'Train loss : {losses.val:.4f}({losses.avg:.4f}) '
        log += f'Top1 : {top1.val / loader.batch_size:.4f}({top1.sum / (i * loader.batch_size):.4f}) '
        log += f'Top5 : {top5.val / loader.batch_size:.4f}({top5.sum / (i * loader.batch_size):.4f})'
        pbar.set_description(log)
        pbar.update()

    _lr = optimizer.param_groups[0]['lr']
    _gap = gap(y_true, y_pred)
    log = f'[EPOCH {epoch}] Train Loss : {losses.avg:.4f}, '
    log += f'Top1 : {top1.sum / loader.dataset.__len__():.4f}, '
    log += f'Top5 : {top5.sum / loader.dataset.__len__():.4f}, '
    log += f'GAP : {_gap:.4e}, '
    log += f'LR : {_lr:.2e}'

    logger.info(log)
    pbar.set_description(log)
    pbar.close()

    writer.add_scalar('Train/Loss', losses.avg, epoch)
    writer.add_scalar('Train/Top1', top1.sum / loader.dataset.__len__(), epoch)
    writer.add_scalar('Train/Top5', top5.sum / loader.dataset.__len__(), epoch)
    writer.add_scalar('Train/GAP', _gap, epoch)
    writer.add_scalar('Train/LR', _lr, epoch)


@torch.no_grad()
def valid(model, loader, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    pbar = tqdm(loader, ncols=150)
    y_true = dict()
    y_pred = dict()


    for i, (image, iid, target, _) in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs = model(image.cuda())
        loss = criterion(outputs, target.cuda())

        conf, indice = torch.topk(outputs, k=5)
        indice = indice.cpu()

        y_true.update({k: t for k, t in zip(iid, target.numpy())})
        y_pred.update({k: (t, c) for k, t, c in
                       zip(iid, indice[:, 0].cpu().detach().numpy(), conf[:, 0].cpu().detach().numpy())})

        top1.update(torch.sum(indice[:, :1] == target.view(-1, 1)).item())
        top5.update(torch.sum(indice == target.view(-1, 1)).item())
        losses.update(loss)

        log = f'[Epoch {epoch}] Valid Loss : {losses.val:.4f}({losses.avg:.4f}), '
        log += f'Top1 : {top1.val / loader.batch_size:.4f}({top1.sum / (i * loader.batch_size):.4f}), '
        log += f'Top5 : {top5.val / loader.batch_size:.4f}({top5.sum / (i * loader.batch_size):.4f})'
        pbar.set_description(log)
        pbar.update()

    _lr = optimizer.param_groups[0]['lr']
    _gap = gap(y_true, y_pred)
    log = f'[EPOCH {epoch}] Valid Loss : {losses.avg:.4f}, '
    log += f'Top1 : {top1.sum / loader.dataset.__len__():.4f}, '
    log += f'Top5 : {top5.sum / loader.dataset.__len__():.4f}, '
    log += f'GAP : {_gap:.4e}'

    logger.info(log)
    pbar.set_description(log)
    pbar.close()

    writer.add_scalar('Valid/Loss', losses.avg, epoch)
    writer.add_scalar('Valid/Top1', top1.sum / loader.dataset.__len__(), epoch)
    writer.add_scalar('Valid/Top5', top5.sum / loader.dataset.__len__(), epoch)
    writer.add_scalar('Valid/GAP', _gap, epoch)


@torch.no_grad()
def test(model, loader, epoch, log_dir):
    model.eval()
    pbar = tqdm(loader, ncols=150)
    iids = []
    classes = []
    confideneces = []
    softmax = nn.Softmax(dim=1)
    for i, (image, iid) in enumerate(loader, start=1):
        outputs = model(image.cuda())
        outputs = softmax(outputs)
        conf, indice = torch.topk(outputs, k=1)
        iids.extend(iid)
        classes.extend(indice[:, 0].cpu().numpy())
        confideneces.extend(conf[:, 0].cpu().numpy())
        pbar.update()

    pbar.close()
    iids = pd.Series(iids, name="id")
    classes = pd.Series(classes, name="landmark_id")
    confideneces = pd.Series(confideneces, name="conf")

    df = pd.concat([iids, classes, confideneces], axis=1)
    df.to_csv(f'{log_dir}/submissions/submission_{model_idx}_{sampling_seed}_ep{epoch:03d}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--category_csv', dest='category_csv',
                        default="/data/public/category.csv")
    parser.add_argument('--train_dir', dest='train_dir',
                        default="/data/public/train/")
    parser.add_argument('--train_csv', dest='train_csv',
                        default="/data/public/train.csv")
    parser.add_argument('--test_dir', dest='test_dir',
                        default="/data/public/test")
    parser.add_argument('--submission_csv', dest='submission_csv',
                        default="/data/public/sample_submission.csv")
    # Log/Ckpt directory
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default="/mldisk/nfs_shared_/dh/landmark_dacon_ckpt/")
    parser.add_argument('--comment', dest='comment', type=str, default=None)

    # Hyper-parameter
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)

    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3)
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0)

    parser.add_argument('-step', '--step_size', type=int, default=5)
    parser.add_argument('-gamma', '--step_gamma', type=float, default=0.8)

    args = parser.parse_args()

    log_dir = init_logger(args.ckpt_dir, args.comment)
    logger.info(args)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    global model_idx
    global sampling_seed

    total_epoch = 0

    model_list = [8]  # 0: b0, 1: b1, 2:b2, 3:b3, 4:b4, 5:Vit-L/16
    sampling_seed_list = [21,22,23,24,25]
    for i in model_list:
        model_idx = i
        model_str = {0: 'b0', 1: 'b1', 2: 'b2', 3: 'b3', 4: 'b4',
                     5: 'seresnext50_32x4d', 6: 'seresnext101_32x4d',
                     7: 'vit_base_patch16_224', 8: 'vit_small_patch16_224'}[model_idx]
        args.batch_size = {0: 128, 1: 128, 2: 64, 3: 32, 4: 32, 5: 128, 6: 64, 7: 64, 8: 192}[model_idx]
        input_size = {0: 224, 1: 240, 2: 260, 3: 300, 4: 380, 5: 224, 6: 224, 7: 224, 8: 224}[model_idx]
        model_ckpt = '/data/ckpt_epoch_4_039.pt'
        model_load = False
        if model_idx >= 0 and model_idx <= 4:
            model = Efficientnet(model_idx).cuda()

        if model_idx == 5:
            model = gluon_seresnext50_32x4d(pretrained=True, num_classes=1049)
        if model_idx == 6:
            model = gluon_seresnext101_32x4d(pretrained=True, num_classes=1049)
        if model_idx == 7:
            model = vit_base_patch16_224(pretrained=True, num_classes=1049)
        if model_idx == 8:
            model = vit_small_patch16_224(pretrained=True, num_classes=1049)
            # freeze layer
            # grad = False
            # for n, p in model.named_parameters():
            #     p.requires_grad = grad = grad or n.startswith('base._blocks.22')
            #     logger.info(f'Layer : {n},  Grad : {p.requires_grad}')

        if model_load:
            model_dict = torch.load(model_ckpt)['model_state_dict']
            model.load_state_dict(model_dict)
            print('{}} was loaded. '.format(model_ckpt))
        else:
            print('local ckpt file was not loaded. ')
        try:
            logger.info(model.summary((3, input_size, input_size)))
        except:
            print('model summary is failed')

        model = nn.DataParallel(model.cuda())
        # optimizer = toptim.RAdam(filter(lambda p: p.requires_grad, model.parameters()),
        #                          lr=args.learning_rate,
        #                          betas=(0.9, 0.999),
        #                          weight_decay=args.weight_decay)
        optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.step_gamma)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))

        criterion = nn.CrossEntropyLoss()

        for j in sampling_seed_list:
            sampling_seed = j

            train_trn = trn.Compose([
                trn.RandomResizedCrop(input_size),
                ImageNetPolicy(),
                # trn.Resize((224, 224)),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            valid_trn = test_trn = trn.Compose([
                trn.Resize((input_size, input_size)),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            train_dataset = LandmarkDataset(args.train_dir, args.train_csv, args.category_csv, train_trn, 'train',seed=sampling_seed)
            valid_dataset = LandmarkDataset(args.train_dir, args.train_csv, args.category_csv, valid_trn, 'valid',seed=sampling_seed)
            test_dataset = TestDataset(args.test_dir, args.submission_csv, args.category_csv, test_trn)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


            # import pdb;pdb.set_trace()
            for ep in range(1, args.epochs):
                train(model, train_loader, criterion, optimizer, total_epoch)
                valid(model, valid_loader, criterion, total_epoch)
                test(model, test_loader, total_epoch, log_dir)
                scheduler.step()

                torch.save({'model_state_dict': model.module.state_dict(),
                            'optim_state_dict': optimizer.state_dict(),
                            'epoch': total_epoch, },
                           f'{log_dir}/ckpts/ckpt_epoch_{model_idx}_{sampling_seed}_{total_epoch:03d}.pt')
                total_epoch +=1

            import gc
            gc.collect()
