import time
import datetime
import os
import os.path as osp
import torch
import torchvision
import torch.distributed as dist
import logging
from collections import defaultdict, deque
from functools import partial
from matplotlib.transforms import Bbox
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.rcParams['font.weight'] = 'bold'

from dataset import BaseDataset


def setup_logger(log_path=None, log_level=logging.INFO):
    logger = logging.root
    logger.setLevel(log_level)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    if log_path is not None:
        log_file = os.path.join(log_path, "log.txt")
        os.makedirs(log_path, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=None, fmt=None):
        if fmt is None:
            fmt = "{value:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    @property
    def all(self):
        return list(self.deque)

    def __str__(self):
        return self.fmt.format(median=self.median,
                               avg=self.avg,
                               global_avg=self.global_avg,
                               max=self.max,
                               value=self.value)


class MetricLogger(object):

    def __init__(self, logger, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'iter time: {time}', 'data time: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'iter time: {time}', 'data time: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            #定义一个生成器函数
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    self.logger.info(
                        log_msg.format(i,
                                       len(iterable),
                                       eta=eta_string,
                                       meters=str(self),
                                       time=str(iter_time),
                                       data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class InterLayerHook:

    def __init__(self, model, layer_name, device):
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.data = None

    def __enter__(self):
        layer = getattr(self.model, self.layer_name)

        def output_hook(module, args, output, obj=None):
            obj.data = output.detach().clone().to(obj.device)

        self.handle = layer.register_forward_hook(
            partial(output_hook, obj=self))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handle.remove()


def draw_bar(values, label_names, save_file, title, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 8))  # 可以设置图形的大小
    x = np.arange(len(values))
    bars = ax.bar(x, values, **kwargs)
    ax.set_xticks(x, BaseDataset.ac_codes, rotation=45)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel('Pattern', font={'size': 16})
    ax.set_ylabel(title, font={'size': 16})

    # 在每个柱状图上显示值
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, '{:.2f}'.format(yval), ha='center', va='bottom')
    fig.tight_layout()
    plt.savefig(save_file)
    plt.close()


class DrawLabelWisePerformance:
    def __init__(self, label_names, save_dir):
        self.label_names = label_names
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.num_classes = len(label_names)
        
        self.preds = []
        self.targets = []
        self.matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    def update(self, predicts, labels):
        self.preds.append(predicts.cpu())
        self.targets.append(labels.cpu())

    def saveMatrix(self, title):
        np.save(osp.join(self.save_dir, 'confusion_matrix_{}.npy'.format(title)), self.matrix)

    def draw(self):

        self.preds = torch.cat(self.preds, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

        acc = (self.preds == self.targets).sum(dim=0) / self.preds.shape[0]

        denominator = self.preds.sum(dim=0).float()
        denominator[denominator == 0] = float("inf")
        prec = (self.preds * self.targets).sum(dim=0) / denominator

        denominator = self.targets.sum(dim=0).float()
        denominator[denominator == 0] = float("inf")
        recall = (self.preds * self.targets).sum(dim=0) / denominator

        fscore = []
        for p, r in zip(prec, recall):
            if p + r == 0:
                fscore.append(0)
            else:
                fscore.append(2 * p * r / (p + r))
        fscore = torch.tensor(fscore)

        draw_bar(acc, self.label_names, osp.join(self.save_dir, 'acc.pdf'), "Accuracy")
        draw_bar(prec, self.label_names, osp.join(self.save_dir, 'precision.pdf'), "Precision", color=(255/255, 159/255, 127/255))
        draw_bar(recall, self.label_names, osp.join(self.save_dir, 'recall.pdf'), "Recall", color=(50/255, 196/255, 233/255))
        draw_bar(fscore, self.label_names, osp.join(self.save_dir, 'fscore.pdf'), "F-score")
        
        for i in range(self.preds.shape[0]):
            pred = self.preds[i]
            target = self.targets[i]
            for j in range(pred.shape[0]):
                if pred[j]:
                    self.matrix[j] += target
        norm_matrix = self.matrix.float() / self.preds.sum(dim=0)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        fontsize = 12
        ax.imshow(norm_matrix, cmap=plt.cm.Blues)
        ax.set_xlabel('Prediction', font={'size': 18})
        ax.set_ylabel('True', font={'size': 18})
        ax.set_xticks(range(norm_matrix.shape[0]), self.label_names, rotation=60)
        ax.set_yticks(range(norm_matrix.shape[1]), self.label_names)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        for i in range(norm_matrix.shape[0]):
            for j in range(norm_matrix.shape[1]):
                ax.text(i, j, f"{norm_matrix[i][j]:.2f}", verticalalignment='center', horizontalalignment='center')
        fig.tight_layout()
        plt.savefig(osp.join(self.save_dir, 'confusion_matrix.pdf'), bbox_inches="tight")
        plt.close()

        return acc, prec, recall, fscore


def draw_bar_title(values, label_names, title, save_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(values))
    bars = ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_title(title)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, '{:.2f}'.format(yval), ha='center', va='bottom')

    plt.savefig(save_file)
    plt.close()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()