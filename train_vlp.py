
# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig

# *user-defined
from models import  SLRCLIP
import utils as utils
from datasets import S2T_Dataset

# *basic
import gc
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import wandb
import copy
from pathlib import Path
import math
import sys
from typing import Iterable, Optional
from loguru import logger


# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER

# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from timm.loss import SoftTargetCrossEntropy

# visualization
from torchvision.utils import save_image, make_grid
from PIL import Image


from hpman.m import _
import hpargparse

# global definition
from definition import *
import pickle
def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.
    :param seed: random seed
    """
    torch.manual_seed(seed) #设置 Pytorch 的随机数生成器的种子。
    torch.cuda.manual_seed(seed) #设置 CUDA 的随机数生成器的种子。
    torch.cuda.manual_seed_all(seed) #设置所有 CUDA 设备的随机数生成器的种子。
    np.random.seed(seed) #设置 NumPy 的随机数生成器的种子。
    random.seed(seed) #设置 Python 的随机数生成器的种子。
    os.environ['PYTHONHASHSEED'] = str(seed) #设置了 Python 的哈希种子，以确保字典和集合的可重复性。
    torch.backends.cudnn.benchmark = False #关闭了 cuDNN 的基准测试模式，以确保每次运行时都使用相同的算法
    torch.backends.cudnn.deterministic = True #设置了 cuDNN 为确定性模式，以确保每次运行时都获得相同的结果。

def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) scripts', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default=0, type=int)#命令行参数加进来的
    # parser.add_argument('--nproc_per_node', default=1, type=int)#命令行参数加进来的

    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,  
    #                     help='number of distributed processes') #指定分布式训练过程中总共有多少个进程
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training') #用于设置分布式训练的 URL，通常设置为 env://，表示使用环境变量中的 URL
    # parser.add_argument('--local_rank', default=0, type=int) #指定当前进程的全局排名，用于在分布式训练中初始化进程


    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint') #微调的预训练模型检查点的路径

    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"') #指定优化器的类型，默认为 "adamw"
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)') #优化器的 epsilon 参数，默认为 1.0e-09
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', 
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)') #优化器的 beta 参数，默认为 [0.9, 0.98]
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)') #指定梯度裁剪的 norm，默认为 None（不裁剪）。
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)') #SGD 动量的参数，默认为 0.9
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)') #权重衰减参数，默认为 0.05

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"') #学习率调度器的类型，默认为 "cosine"
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)') #初始学习率，默认为 1.0e-3
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages') #用于控制学习率噪音的百分比
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)') #用于控制学习率噪音的百分比上限，默认为 0.67
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)') #学习率噪声的标准差（默认为 1.0
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)') #预热阶段的学习率，默认为 1e-6
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)') #循环调度器达到 0 时的下界学习率（默认为 1.0e-8）。
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR') #学习率衰减的周期数
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports') #学习率预热的周期数，如果调度器支持的话。
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends') #学习率冷却的周期数，当循环调度结束时，会在最小学习率处停留一段时间
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10') #指定耐心周期数，用于 Plateau 学习率调度器（默认为 10）
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)') #指定学习率衰减率（默认为 0.1）
    
     # * Baise params
    parser.add_argument('--output_dir', default='out/vlp1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint') # 恢复训练，从检查点开始
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',   #指定从哪个周期开始训练模型
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only') # 只进行模型评估，不进行训练
    parser.add_argument('--num_workers', default=4, type=int) # 用于数据加载的子进程数量。
    parser.add_argument('--pin-mem', action='store_true', # 用于在数据加载过程中将数据移至GPU上，是否固定 CPU 内存。
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.') 
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='') #不固定 CPU 内存
    parser.set_defaults(pin_mem=True) #设置成fasle看看能不能减少内存
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free.yaml') #用于指定模型的超参数和其他配置信息。

    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    
    # * wandb params，用于配置 Wandb 日志记录
    parser.add_argument("--log_all", action="store_true",
        help="flag to log in all processes, otherwise only in rank0",
    ) #一个布尔值标志，指定是否在所有进程中记录日志，或者只在 rank 0 进程中记录日志
    parser.add_argument("--entity", type=str, 
        help="wandb entity",
    ) #指定 Wandb 实体的字符串
    parser.add_argument("--project", type=str, default='VLP',
        help="wandb project",
    ) #指定 Wandb 项目的字符串，默认值为 'VLP'。

    return parser

class CrossEn11(nn.Module):
    """cross entroy loss"""
    def __init__(self,path):
        super(CrossEn11, self).__init__()
        self.eps = 0.2
        self.adaptive_training=True
        self.padding_idx=1
        self.adaptive_method ='exp1234'
        self.adaptive_T=1.75
        if self.adaptive_training:
            self.weight_drop = 0.3
            # 加载词频字典  
            with open('word_freq_dict.pkl', 'rb') as f:  
                self.loaded_word_freq_dict = pickle.load(f) 
            # self.loaded_word_freq_dict=loaded_word_freq_dict.cuda()
            #已经试过6,10,20,50,100
            self.mid=10
            # 计算文本序列的高斯权重  
            # all_freqs = list(self.loaded_word_freq_dict.values())  
            # mean_freq = np.mean(all_freqs)  
            # std_dev_freq = np.std(all_freqs) 
            # freq = []
            # for x, y in read2columns(args.dict_file, split=' '):
            #     freq.append(int(y))
            # freq = torch.tensor(freq)
            # mid = freq[int(len(freq) / 2)]
            # if args.adaptive_method is 'exp':
            #     self.weight = [torch.exp(-1 * args.adaptive_T * item / mid) for item in freq]
            #     b = self.weight.max()
            #     self.weight = self.weight / b * (np.e - 1) + 1
            # else:
            #     self.weight = [torch.pow(item / mid, torch.tensor(2)) * torch.exp(-1 * args.adaptive_T * item / mid) for item in freq]
            #     b = self.weight.max()
            #     self.weight = self.weight / b * (np.e - 1) + 1
            # self.weight = torch.cat([torch.tensor([1., 1., 1., 1.]), self.weight], dim=0)


    def compute_loss(self, logits, label, reduce=False):
        lprobs =F.log_softmax(logits, dim=-1)
        lprobs=lprobs.view(-1, lprobs.size(-1))
        target = label
        non_pad_mask = target.ne(self.padding_idx)
        # 根据 target 张量中的键从字典中取出对应的值，并形成相同尺寸的一维张量  
        weight=[self.loaded_word_freq_dict[key] for key in target.tolist()]
        if self.adaptive_method =='exp':
            loss_weight = [torch.exp(-1 * self.adaptive_T * torch.tensor(item)/ self.mid) for item in weight]
            b = max(loss_weight)
            loss_weight = torch.tensor(torch.tensor(loss_weight) / b * (np.e - 1) + 1).cuda()
        elif self.adaptive_method=="exp123":
            loss_weight = [torch.pow(torch.tensor(item) / self.mid, torch.tensor(2)) * torch.exp(-1 * self.adaptive_T * torch.tensor(item) / self.mid) for item in weight]
            b = max(loss_weight)
            loss_weight = torch.tensor(torch.tensor(loss_weight) / b * (np.e - 1) + 1).cuda()
        else:
            loss_weight = [torch.pow(torch.tensor(item) / self.mid, torch.tensor(2)) * torch.exp(-1 * self.adaptive_T * torch.tensor(item) / self.mid)* torch.exp(-0.5 *  torch.pow(torch.tensor(item-90), torch.tensor(2))/torch.pow(torch.tensor(370.0), torch.tensor(2))) for item in weight]
            b = max(loss_weight)
            loss_weight = torch.tensor(torch.tensor(loss_weight) / b * (np.e - 1) + 1).cuda()

        drop_p = self.weight_drop * torch.ones_like(loss_weight)
        drop_mask = torch.bernoulli(drop_p).byte()
        loss_weight.masked_fill_(drop_mask, 1.)
        nll_loss = -(loss_weight * (lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        else:
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
    def forward(self, logits, label):
        loss, nll_loss = self.compute_loss(logits, label, reduce=False)
        return loss
    
def main(args, config):
    utils.init_distributed_mode(args) #初始化分布式训练模式，如果需要，则设置`args.distributed`为True
    print(args)

    device = torch.device(args.device) #设置训练设备，如果是多卡训练，则使用`torch.device('cuda')`

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank() #设置随机数种子，在每个进程中增加随机数种子，以便在不同的进程中生成不同的随机数。
    # seed = args.seed
    # torch.manual_seed(seed) #设置PyTorch的随机数种子
    # np.random.seed(seed) #设置NumPy的随机种子
    # random.seed(seed) #设置Python的随机种子
    # cudnn.benchmark = False # Since the input dim is dynamic.设置cuDNN的benchmark为False，因为输入维度是动态的。

    set_seed(seed=args.seed)

    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['transformer']) #使用预训练的模型初始化tokenizer

    train_data = S2T_Dataset(path=config['data']['train_label_path'],path1=config['data']['train_head_rgb_input'], tokenizer = tokenizer, config=config, args=args, phase='train')
    print(train_data) #path是训练标签文件的路径，tokenizer是分词器，config是配置文件，args是命令行参数，phase是数据集的阶段，这里是训练阶段
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)#分布式采样
    train_sampler=torch.utils.data.RandomSampler(train_data)#普通的随机采样
    # train_sampler=None
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, #用于数据加载的子进程数量。
                                 collate_fn=train_data.collate_fn, #用于合并样本的函数
                                 sampler=train_sampler, #用于分配样本的采样器
                                 pin_memory=args.pin_mem, #是否使用固定内存，将数据移至GPU上
                                 drop_last=True) #是否在最后一个小批次中丢弃数据
    
    
    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], path1=config['data']['dev_head_rgb_input'],tokenizer = tokenizer, config=config, args=args, phase='val')
    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_sampler=torch.utils.data.RandomSampler(dev_data)
    # dev_sampler=None
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)

    test_data = S2T_Dataset(path=config['data']['test_label_path'],path1=config['data']['test_head_rgb_input'], tokenizer = tokenizer, config=config, args=args, phase='test')
    print(test_data)
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler=torch.utils.data.RandomSampler(test_data)
    # test_sampler=None
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)
    
    print(f"Creating model:")
    model = SLRCLIP(
        config=config
        )
    model.to(device)
    print(model)

    if args.finetune:#一个预训练模型的基础上，对模型进行进一步的训练(微调)
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret =  model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))#打印出在加载模型参数时，缺失的和意外的参数
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model#创建一个模型副本，用于在分布式环境中运行，而原始模型用于存储和更新
    if args.distributed: #用于分布式训练
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)#BatchNorm层转换为同步BatchNorm层，以支持分布式训练
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    # create optimizer and scheduler
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = utils.KLLoss()
    loss_scaler = NativeScaler()

    output_dir = Path(args.output_dir)
    if args.resume: #这段代码用于从检查点中恢复训练，使得在中断的训练后可以继续进行训练
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])#检查是否需要恢复优化器、学习率调度器和训练进度，非评估模式
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1 #从检查点中加载训练进度，并将当前训练进度设置为上一个检查点的下一轮

    if args.eval:#用于评估模型在验证集和测试集上的性能，并打印损失。在评估之前，需要指定训练过的模型的检查点
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, dev_dataloader, model, criterion, args.start_epoch)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, criterion, args.start_epoch)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    print(f"Start training for {args.epochs} epochs")#训练模型，并在每个周期结束时更新学习率、计算训练统计信息，以及保存检查点
    start_time = time.time()
    min_loss = np.inf #初始化最小损失值为正无穷大
    second_loss=1000.10
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:#如果使用分布式训练，设置训练数据采样器的周期
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config, loss_scaler)
        #调用`train_one_epoch`函数训练一个周期，并返回训练统计信息。
        lr_scheduler.step(epoch)#更新学习率

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:#循环保存检查点文件。
                utils.save_on_master({#在主进程中保存检查点
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        test_stats = evaluate(args, dev_dataloader, model, criterion, epoch)#评估模型在验证集上的表现

        if min_loss > test_stats["loss"]:#更新并检查最小验证损失
            min_loss = test_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / f'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'args': args,
                    }, checkpoint_path)
        elif min_loss < test_stats["loss"] <second_loss:
            second_loss = test_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / f'second_best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'args': args,
                    }, checkpoint_path)
        
        print(f"* DEV loss {test_stats['loss']:.3f} Min DEV loss {min_loss}")#打印验证损失
        if utils.is_main_process():# 如果是主进程，记录并上传数据到Wandb
            wandb.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'dev/dev_loss':test_stats['loss'], 'dev/min_loss': min_loss})

        #准备日志统计数据
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        #如果args.output_dir为True，表示要保存日志，那么将log_stats以JSON格式写入到指定的"log.txt"文件中
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        gc.collect()#执行垃圾回收
        if hasattr(torch.cuda, 'empty_cache'):#如果torch.cuda具有'empty_cache'属性，清空CUDA缓存。
            torch.cuda.empty_cache()
    # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        # torch.distributed.barrier() #确保所有进程都在此行同步
        checkpoint = torch.load(args.output_dir+'/best_checkpoint.pth', map_location='cpu')#指定路径加载检查点，并将其映射到CPU
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)#将检查点中的模型状态加载到`model_without_ddp`中，严格匹配权重参数

        dev_stats = evaluate(args, dev_dataloader, model, criterion, epoch)#评估验证集
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, criterion, epoch)#评估测试集
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
    if args.output_dir:
         # torch.distributed.barrier() #确保所有进程都在此行同步
        checkpoint = torch.load(args.output_dir+'/second_best_checkpoint.pth', map_location='cpu')#指定路径加载检查点，并将其映射到CPU
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)#将检查点中的模型状态加载到`model_without_ddp`中，严格匹配权重参数

        dev_stats = evaluate(args, dev_dataloader, model, criterion, epoch)#评估验证集
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, criterion, epoch)#评估测试集
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))#打印训练时间

def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)#将模型设置为训练模式

    metric_logger = utils.MetricLogger(delimiter="  ")#初始化一个`MetricLogger`，用于记录训练过程中的各种指标
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))#添加一个记录学习率的指标，使用`SmoothedValue`来平滑学习率的变化
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)#表示当前epoch的信息
    print_freq = 10 #每10个数据批次打印一次日志
    loss_img = criterion
    loss_txt = criterion
    # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)
    loss_fct=CrossEn11('word_freq_dict.pkl')
    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    #使用`enumerate`和`metric_logger.log_every`遍历数据加载器中的每一个数据批次
        optimizer.zero_grad()#在每次迭代前，清空梯度
        with torch.cuda.amp.autocast():#用模型`model`进行前向传播，得到`logits_per_image`、`logits_per_text`和`ground_truth`
            logits_per_image, logits_per_text, ground_truth,lm_logits,m_loss= model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image,ground_truth)#计算图片和文本的损失，分别是`loss_imgs`和`loss_texts`，然后计算总损失`total_loss`。
            loss_texts = loss_txt(logits_per_text,ground_truth)
            total_loss=(loss_imgs + loss_texts)/2.
            # pearson_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].cuda().view(-1)) 
            pearson_loss = loss_fct(lm_logits, tgt_input['input_ids'].cuda().view(-1)) 
            total_loss1 = total_loss+pearson_loss+m_loss*0.1
        loss_scaler(total_loss1, optimizer)#使用`loss_scaler`对总损失和优化器进行缩放。
        


        loss_value = total_loss.item()
        if not math.isfinite(loss_value):#如果损失值不是有限的，打印损失值并停止训练。
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)#更新`MetricLogger`中的损失和学习率。
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(clip_loss=clip_loss.item())
        metric_logger.update(pearson_loss=pearson_loss.item())
        metric_logger.update(m_loss=m_loss.item())
        metric_logger.update(total_loss1=total_loss1.item())

        if (step+1) % 10 == 0 and utils.is_main_process():
            #如果当前步骤是10的倍数，并且在主进程中，则对`logits_per_image`和`logits_per_text`进行拼接，并调用`utils.visualization`进行可视化
            visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
            # utils.visualization([visual_map,])

    if args.run:#记录当前epoch和训练损失到日志中
        args.run.log({'epoch':epoch+1,'epoch/train_loss':loss_value})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()#将所有进程的统计数据同步到主进程中
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}#返回一个字典，其中键是各个指标的名称,返回训练过程中的平均损失

def evaluate(args, dev_dataloader, model, criterion, epoch):
    model.eval()#将模型设置为评估模式

    metric_logger = utils.MetricLogger(delimiter="  ") #用于记录评估过程中的各种指标
    header = 'Test:' #设置日志的标题为Test
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)
    with torch.no_grad():#不需要计算梯度
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, print_freq, header)):

            logits_per_image, logits_per_text, ground_truth,lm_logits,m_loss= model(src_input, tgt_input)#前向传播，得到图像和文本的logits以及真实的标签
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            total_loss = (loss_imgs + loss_texts)/2
            pearson_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].cuda().view(-1)) 

            metric_logger.update(loss=total_loss.item())#更新`metric_logger`，记录损失
            # metric_logger.update(clip_loss=clip_loss.item())
            metric_logger.update(pearson_loss=pearson_loss.item())
            metric_logger.update(m_loss=m_loss.item())
            metric_logger.update(total_loss1=(total_loss.item()+pearson_loss.item()+m_loss.item()*0.1))

            if (step+1) % 10 == 0 and utils.is_main_process():#当前批次数能被10整除，并且在主进程中，则进行可视化
                visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
                # utils.visualization([visual_map, ])

    if args.run: #如果`args.run`存在，则记录当前epoch和验证集上的损失到日志中
        args.run.log({'epoch':epoch+1,'epoch/dev_loss':total_loss.item()})
    metric_logger.synchronize_between_processes()#同步所有进程中的统计数据
    print("* Averaged stats:", metric_logger)#打印同步后的统计数据
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}#返回一个字典，其中键是各个指标的名称，值是这些指标在全局范围内的平均值

def setup_run(args, config):#用来设置Wandb的运行记录
    if args.log_all:#如果args.log_all为True，那么在每个进程中初始化一个Wandb的运行记录
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled' #根据是否为评估模式来设置Wandb的模式
        run = wandb.init( #初始化Wandb运行记录，并设置相应的参数
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch") #定义一个名为"epoch"的指标
        run.define_metric("training/*", step_metric="epoch") #定义一个名为"training/*"的指标，其步长为"epoch"
        run.define_metric("dev/*", step_metric="epoch") #定义一个名为"dev/*"的指标，其步长为"epoch"
    else:
        if utils.is_main_process():#如果`args.log_all`为False，则只记录主进程中的运行信息
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            run.name = args.output_dir.split('/')[-1] #设置Wandb运行记录的名称为输出目录的名称
        else:
            os.environ["WANDB_MODE"] = 'disabled' #设置Wandb的模式为disabled
            run = False 

    return run
if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false" # 设置环境变量为`false`，设置huggingface的tokenizers库的并行设置为false

    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) scripts', parents=[get_args_parser()])#创建一个命令行参数解析器
    _.parse_file(Path(__file__).resolve().parent) #解析当前文件的父目录中的配置文件
    hpargparse.bind(parser, _)#将命令行参数解析器与配置文件中的参数进行绑定
    args = parser.parse_args()#解析命令行参数


    with open(args.config, 'r+',encoding='utf-8') as f: #以读取模式打开配置文件
        config = yaml.load(f,Loader=yaml.FullLoader) #使用yaml.FullLoader加载配置文件
    
    # wandb.init a run if logging, otherwise return None
    args.run = setup_run(args, config)#设置Wandb的运行记录
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)#创建输出目录
    # gc.collect()#执行垃圾回收
    # if hasattr(torch.cuda, 'empty_cache'):#如果torch.cuda具有'empty_cache'属性，清空CUDA缓存。
    #     torch.cuda.empty_cache()
    main(args, config)