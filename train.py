import os
import os.path as osp
import json
import argparse
import time
import datetime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision

from utils import MetricLogger, setup_logger, init_distributed_mode, DrawLabelWisePerformance, is_main_process

import numpy as np
from dataset import Anahep2DatasetTrain, Anahep2DatasetEval, eval_collate_fn
from torch.utils.tensorboard import SummaryWriter


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    # dataset
    parser.add_argument('-d', '--data_path', metavar='DIR', default="/path/to/crops",
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--label_file', metavar='DIR', default="/path/to/splits.json",
                        help='class label file')
    parser.add_argument('--img_size', default=224, type=int,
                        help="input image shape")
    
    # model
    parser.add_argument("--model", default="resnet50", type=str, help="model name")

    # training
    parser.add_argument('--test', action='store_true',
                        help='test only')
    parser.add_argument("--sync_bn", action="store_true",
                        help="Use sync batch norm")
    parser.add_argument("--device", default="cuda", type=str, 
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='images per gpu')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument("--lr_warmup_epochs", default=5, type=int, 
                        help="the number of epochs to warmup (default: 0)")
    parser.add_argument('--wd', '--weight_decay', default=2e-5, type=float,
                        metavar='WD', help='weight decay rate', dest='wd')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--epochs', default=100, type=int,
                        metavar='N', help='number of training epochs')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-n', '--name', default="train", type=str, metavar='PATH',
                        help='name of this project')
    parser.add_argument("--world_size", default=1, type=int, 
                        help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", type=str, 
                        help="url used to set up distributed training")
    
    return parser


def main(args):
    args.ckpt_dir = osp.join("checkpoints", args.name)

    init_distributed_mode(args)

    if not args.test and is_main_process():
        setup_logger(log_path=args.ckpt_dir)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(osp.join(args.ckpt_dir, "config.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)
    else:
        setup_logger(log_path=None)
        if args.resume and osp.isfile(args.resume):
            args.ckpt_dir = osp.dirname(args.resume)
    
    logger = logging.getLogger("Main")

    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    tb_writer = SummaryWriter(log_dir=osp.join("tb", args.name))

    cudnn.benchmark = True
    device = torch.device(args.device)

    train_dataset = Anahep2DatasetTrain(
        args.data_path,
        args.label_file,
        image_size=args.img_size
    )
    test_dataset = Anahep2DatasetEval(
        args.data_path,
        args.label_file,
        split="test",
        image_size=args.img_size
    )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        pin_memory=True,
        collate_fn=eval_collate_fn,
        num_workers=args.workers
    )
    model = torchvision.models.get_model(args.model, weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=train_dataset.num_labels)
    model = model.to(device)
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    optimizer = torch.optim.Adam(
        parameters, args.lr,
        weight_decay=args.wd
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs - args.lr_warmup_epochs, eta_min=1e-6
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.lr_warmup_epochs
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, lr_scheduler], milestones=[args.lr_warmup_epochs]
    )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.test and is_main_process():
        cudnn.benchmark = False
        cudnn.deterministic = True
        logger.info(' ---- test model ---- ')
        eval(test_loader, model_without_ddp, criterion, device, args, "Test")
        return
    
    logger.info(' ---- train model ---- ')
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args, tb_writer)
        lr_scheduler.step()

        if is_main_process():
            checkpoint = {
                'epoch': epoch,
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.ckpt_dir, "checkpoint.pt"))

            # evaluate on test set
            acc, p, r, f_score = eval(test_loader, model_without_ddp, criterion, device, args, "Test")

            tb_writer.add_scalar(tag='test_acc', scalar_value=acc, global_step=epoch)
            tb_writer.add_scalar(tag='test_precision', scalar_value=p, global_step=epoch)
            tb_writer.add_scalar(tag='test_recall', scalar_value=r, global_step=epoch)
            tb_writer.add_scalar(tag=f'test_f_score', scalar_value=f_score, global_step=epoch)

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def train(train_loader, model, criterion, optimizer, epoch, device, args, writer):
    logger = logging.getLogger("Train")

    metric_logger = MetricLogger(logger, delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for i, batch in enumerate(metric_logger.log_every(train_loader, args.print_freq, header=header)):
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # compute output
        output = model(images)
        loss = criterion(output, targets)
        acc = accuracy(output, targets)

        metric_logger.update(loss=loss, acc=acc, n=images.size(0))

        # tensorboard
        if is_main_process():
            writer.add_scalar(tag='train_loss', scalar_value=loss, global_step=len(train_loader)*epoch+i)
            writer.add_scalar(tag='acc', scalar_value=acc, global_step=len(train_loader)*epoch+i)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


@torch.inference_mode()
def eval(val_loader, model, criterion, device, args, header="Eval"):
    logger = logging.getLogger(header)

    metric_logger = MetricLogger(logger, delimiter="  ")
    batch_size = args.batch_size
    # switch to evaluate mode
    model.eval()

    draw = DrawLabelWisePerformance(val_loader.dataset.pattern_names, osp.join(args.ckpt_dir, "viz"))

    # label_studio_ann = []

    for _, batch in enumerate(metric_logger.log_every(val_loader, args.print_freq, header=header)):
        images, targets, num_crops, img_ids = batch
        targets = targets.to(device, non_blocking=True)

        outputs = []
        for i in range(0, images.shape[0], batch_size):
            batch_images = images[i:i+batch_size].to(device, non_blocking=True)
            batch_targets = targets[i:i+batch_size]

            # compute output
            output = model(batch_images)
            loss = criterion(output, batch_targets)
            output = F.sigmoid(output)
            outputs.append(output)

            metric_logger.update(loss=loss, n=batch_images.shape[0])

        outputs = torch.cat(outputs, dim=0)

        image_outputs = []
        image_targets = []
        i = 0
        for num_crop in num_crops:
            # average
            image_outputs.append(outputs[i:i+num_crop].mean(dim=0))
            # union
            image_outputs.append(outputs[i:i+num_crop].max(dim=0)[0])
            # voting
            # pred = (outputs[i:i+num_crop] >= 0.5).float().mean(dim=0)
            # image_outputs.append(pred)
            image_targets.append(targets[i])
            i += num_crop
        image_outputs = torch.stack(image_outputs, dim=0)
        image_targets = torch.stack(image_targets, dim=0)

        draw.update((image_outputs>=0.5).long(), image_targets.long())
        acc = accuracy(image_outputs, image_targets, 0.5)
        p = precision(image_outputs, image_targets, 0.5)
        r = recall(image_outputs, image_targets, 0.5)
        metric_logger.update(acc=acc, p=p, r=r, n=image_outputs.shape[0])
    
    mc_acc, mc_p, mc_r, mc_f = draw.draw()
    logger.info(f"mean-class acc: {mc_acc.mean()}")
    logger.info(f"mean-class precision: {mc_p.mean()}")
    logger.info(f"mean-class recall: {mc_r.mean()}")
    logger.info(f"mean-class f-score: {mc_f.mean()}")

    logger.info(f"image-level acc: {metric_logger.acc.global_avg}")
    logger.info(f"image-level precision: {metric_logger.p.global_avg}")
    logger.info(f"image-level recall: {metric_logger.r.global_avg}")
    if metric_logger.p.global_avg + metric_logger.r.global_avg == 0:
        f_score = 0
    else:
        f_score = 2 * metric_logger.p.global_avg * metric_logger.r.global_avg / (metric_logger.p.global_avg + metric_logger.r.global_avg)
    logger.info(f"image-level f-score: {f_score}")

    with open(os.path.join(args.ckpt_dir, "metrics.json"), "w") as f:
        json.dump({
            "mean_class_acc": mc_acc.tolist(),
            "mean_class_precision": mc_p.tolist(),
            "mean_class_recall": mc_r.tolist(),
            "mean_class_f_score": mc_f.tolist(),
            "average_acc": metric_logger.acc.global_avg,
            "average_precision": metric_logger.p.global_avg,
            "average_recall": metric_logger.r.global_avg,
            "average_f_score": f_score
        }, f, indent=4)

    return metric_logger.acc.global_avg, metric_logger.p.global_avg, metric_logger.r.global_avg, f_score


@torch.inference_mode()
def accuracy(output, target, boundary=0.5):
    pred = output >= boundary
    acc = torch.logical_and(pred, target).sum(dim=1).float() / torch.logical_or(pred, target).sum(dim=1).float()
    acc = acc.mean()

    return acc

@torch.inference_mode()
def precision(output, target, boundary=0.5):
    pred = output >= boundary
    num_samples = pred.shape[0]

    mask = pred.any(dim=1)
    index = torch.arange(num_samples, device=pred.device)[mask]

    pred = pred.index_select(0, index)
    target = target.index_select(0, index)
    p = torch.logical_and(pred, target).sum(dim=1).float() / pred.sum(dim=1).float()
    p = p.sum() / num_samples

    return p

@torch.inference_mode()
def recall(output, target, boundary=0.5):
    pred = output >= boundary
    r = torch.logical_and(pred, target).sum(dim=1).float() / target.sum(dim=1).float()
    r = r.mean()

    return r


if __name__ == "__main__":
    parser = get_parser(description="ANA HEp-2 image classification.")
    args = parser.parse_args()
    main(args)