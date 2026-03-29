
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
from models import  SLRCLIP, Text_Decoder
import utils as utils
from datasets import S2T_Dataset

# *basic
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
from timm.optim import AdamW

# visualization
from torchvision.utils import save_image, make_grid
from PIL import Image

from hpman.m import _
import hpargparse


# global definition
from definition import *
import signcl as signcl
# cl_criterion = signcl.SignCL()

# ========= 计算成本工具（直接粘贴到 train_slt.py） =========
import time
from contextlib import contextmanager
from typing import Any, Mapping, Sequence, Tuple, Dict, Optional, Union

import torch
# 可选依赖：fvcore / thop
try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False

try:
    from thop import profile as thop_profile
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

Nested = Union[torch.Tensor, Mapping[str, Any], Sequence[Any]]

def _move_to(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, Mapping):
        return {k: _move_to(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        y = [_move_to(v, device) for v in x]
        return type(x)(y) if isinstance(x, tuple) else y
    return x

def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, Mapping):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None

def _infer_bs(x: Any) -> int:
    t = _first_tensor(x)
    if t is None or t.ndim == 0:
        return 1
    return int(t.shape[0])

def _model_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")

def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _format_number(n: Union[int, float]) -> str:
    units = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]
    for base, suf in units:
        if abs(n) >= base:
            return f"{n/base:.3f}{suf}"
    return f"{n}"

@contextmanager
def _eval_no_grad(model: torch.nn.Module):
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        yield
    model.train(was_training)

def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def compute_flops_easy(
    model: torch.nn.Module,
    example_inputs: Nested,
    device: Optional[torch.device] = None,
) -> int:
    """
    使用 fvcore 优先统计 FLOPs；失败时回退 thop（返回 MACs ≈ FLOPs）。
    example_inputs: 与 forward 对齐的输入，可以是 (src_input, tgt_input) 的 tuple 或 dict 等。
    """
    if device is None:
        device = _model_device(model)
    inputs = _move_to(tuple(example_inputs), device)

    with _eval_no_grad(model):
        # 优先 fvcore（支持复杂输入）
        if _HAS_FVCORE:
            try:
                flops = int(FlopCountAnalysis(model, inputs).total())
                return flops
            except Exception as e:
                print(f"[FLOPs] fvcore 统计失败，回退到 thop。原因: {repr(e)}")

        # 回退 thop
        if _HAS_THOP:
            try:
                if isinstance(inputs, Mapping):
                    macs, _ = thop_profile(model, inputs=(), kwargs=inputs, verbose=False)
                elif isinstance(inputs, (list, tuple)):
                    macs, _ = thop_profile(model, inputs=tuple(inputs), verbose=False)
                else:
                    macs, _ = thop_profile(model, inputs=(inputs,), verbose=False)
                return int(macs)  # 以 MACs 近似 FLOPs
            except Exception as e:
                print(f"[FLOPs] thop 统计失败：{repr(e)}")

    raise RuntimeError("FLOPs 统计失败。请安装并确认 fvcore 或 thop 可用。")

def benchmark_latency(
    model: torch.nn.Module,
    example_inputs: Nested,
    *,
    device: Optional[torch.device] = None,
    warmup: int = 20,
    iters: int = 100,
    amp: bool = False,
    include_data_transfer: bool = True,
) -> Dict[str, float]:
    """
    平均推理时延（ms），默认包含 CPU->GPU 搬运时间；AMP 可选。
    """
    if device is None:
        device = _model_device(model)

    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    def do_forward(b):
        with torch.autocast(device_type=autocast_device_type, enabled=amp, dtype=amp_dtype):
            if isinstance(b, Mapping):
                _ = model(**b)
            elif isinstance(b, (list, tuple)):
                _ = model(*b)
            else:
                _ = model(b)

    # 预热
    with _eval_no_grad(model):
        b = _move_to(example_inputs, device) if include_data_transfer else example_inputs
        for _ in range(max(1, warmup)):
            if not include_data_transfer:
                b = _move_to(b, device)
            do_forward(b)

        # 正式计时
        times = []
        bs = []
        for _ in range(max(1, iters)):
            if include_data_transfer:
                b = _move_to(example_inputs, device)
            _sync_if_needed(device)
            t0 = time.perf_counter()
            do_forward(b)
            _sync_if_needed(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            bs.append(_infer_bs(b))

    import numpy as np
    t = np.array(times, dtype=float)
    bs = np.array(bs, dtype=float)
    return {
        "avg_ms_per_batch": float(t.mean()),
        "avg_ms_per_sample": float((t / np.clip(bs, 1, None)).mean()),
        "p50_ms": float(np.percentile(t, 50)),
        "p90_ms": float(np.percentile(t, 90)),
        "p95_ms": float(np.percentile(t, 95)),
        "num_iters": float(len(t)),
    }

def profile_slt_model(
    model: torch.nn.Module,
    batch_like: Union[Tuple[Any, Any], Mapping[str, Any], torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    warmup: int = 20,
    iters: int = 100,
    amp: bool = False,
    include_data_transfer: bool = True,
    measure_latency: bool = True,
) -> Dict[str, Any]:
    """
    - batch_like: 直接拿 dataloader 里取出的一个 batch，如 (src_input, tgt_input)
    - 自动统计：参数量、FLOPs、（可选）时延
    """
    if device is None:
        device = _model_device(model)

    # 将 (src_input, tgt_input) 保持原结构传入
    example_inputs = batch_like

    total_params = count_params(model, trainable_only=False)
    trainable_params = count_params(model, trainable_only=True)

    # FLOPs（基于该 batch 的真实尺寸）
    flops = compute_flops_easy(model, example_inputs, device=device)

    result = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "params_human": _format_number(total_params),
        "flops": int(flops),
        "flops_human": _format_number(flops),
    }

    if measure_latency:
        lat = benchmark_latency(
            model,
            example_inputs,
            device=device,
            warmup=warmup,
            iters=iters,
            amp=amp,
            include_data_transfer=include_data_transfer,
        )
        result.update(lat)

    return result
# ========= 计算成本工具（结束） =========


def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)


    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--local_rank', default=0, type=int)


    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
     # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free.yaml')

    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    
    # * wandb params
    parser.add_argument("--log_all", action="store_true",
        help="flag to log in all processes, otherwise only in rank0",
    )
    parser.add_argument("--entity", type=str, 
        help="wandb entity",
    )
    parser.add_argument("--project", type=str, default='VLP',
        help="wandb project",
    )

    # * Noise params，用于配置数据增强和噪声注入
    parser.add_argument('--training-refurbish', default=True, type=bool) #是否在训练过程中进行数据增强，默认为 True。
    parser.add_argument('--noise-rate', default=0.15, type=float) #噪声注入的比率，默认为 0.15
    parser.add_argument('--noise-type', default='omit_last', type=str, choices=['omit', 'omit_last']) #噪声注入的类型，可以是 'omit' 或 'omit_last'，默认为 'omit_last'
    parser.add_argument('--random-shuffle', default=False, type=bool) #是否对数据集进行随机洗牌，默认为 False。

    parser.add_argument('--loss-lambda', type=float, default=1.0, metavar='RATE',
                        help='lambda param') #调整损失函数的权重参数，默认为 1.0。

    return parser

def main(args, config):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print(f"Creating model:")
    model = SLRCLIP(config=config)
    model.to(device)
    print(model)

    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['transformer'])

    train_data = S2T_Dataset(path=config['data']['train_label_path'],path1=config['data']['train_head_rgb_input'], tokenizer = tokenizer, config=config, args=args, phase='train', training_refurbish=True,)
    print(train_data)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_sampler=torch.utils.data.RandomSampler(train_data)  
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)
    
    
    dev_data = S2T_Dataset(path=config['data']['dev_label_path'],path1=config['data']['dev_head_rgb_input'], tokenizer = tokenizer, config=config, args=args, phase='val', training_refurbish=True)
    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_sampler=torch.utils.data.RandomSampler(dev_data)  
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)                                                                                                                                               

    test_data = S2T_Dataset(path=config['data']['test_label_path'],path1=config['data']['test_head_rgb_input'], tokenizer = tokenizer, config=config, args=args, phase='test', training_refurbish=True)
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler=torch.utils.data.RandomSampler(test_data)  
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)



    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret =  model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    # 选一个代表性 batch（优先 dev，再退回 train）
    try:
        example_batch = next(iter(dev_dataloader))[:2]
    except Exception:
        example_batch = next(iter(train_dataloader))

    # 注意：这里用 model_without_ddp 做统计，避免 DDP wrapper 干扰
    if utils.is_main_process():
        print("\n[Profiling] Measuring params / FLOPs / latency ...")
        cost_report = profile_slt_model(
            model_without_ddp,
            batch_like=example_batch,         # 直接传 (src_input, tgt_input)
            device=device,
            warmup=30,                        # 可酌情调大，保证稳定
            iters=100,
            amp=True,                        # 若评测/部署时会用 AMP，可改 True
            include_data_transfer=True,       # 是否计入 CPU->GPU 搬运时间
            measure_latency=True,
        )
        print(f"[Cost] Params: {cost_report['params_human']}, "
            f"FLOPs: {cost_report['flops_human']}, "
            f"Latency: {cost_report['avg_ms_per_batch']:.2f} ms/batch "
            f"({cost_report['avg_ms_per_sample']:.2f} ms/sample)")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    text_decoder = Text_Decoder(config).to(device) #创建一个文本解码器模型 text_decoder，并将其转移到指定的设备上。

    if args.distributed: #将 text_decoder 包装在 DistributedDataParallel 中，以便在多个GPU上进行分布式训练。
        text_decoder = torch.nn.parallel.DistributedDataParallel(text_decoder, device_ids=[args.gpu], find_unused_parameters=True)
    # optimizer_td = AdamW(text_decoder.module.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98)) 
    #取消分布式训练后显示没有module,尝试以下代码
    optimizer_td = AdamW(text_decoder.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98)) #用于优化文本解码器的参数，设置了学习率和权重衰减等参数
    lr_scheduler_td = scheduler.CosineAnnealingLR(
                optimizer=optimizer_td,
                eta_min=1e-8,
                T_max=args.epochs,
            ) #创建一个学习率调度器 lr_scheduler_td，使用余弦退火策略，初始化时学习率为 1e-3，最小学习率为 1e-8，并设置最大周期为 args.epochs。
    TD_train_dict = dict(
        optimizer = optimizer_td,
        lr_scheduler = lr_scheduler_td,
        text_decoder = text_decoder
    ) #将优化器、学习率调度器和文本解码器打包成一个字典 TD_train_dict，以便在训练过程中使用

    criterion = utils.KLLoss() #定义一个 KL 散度损失函数 criterion
    loss_scaler = NativeScaler() #损失缩放器可以帮助稳定训练过程。原生缩放器是一种简单的缩放方法，它将损失值乘以一个缩放因子，以使其在适当的范围内。

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, args.start_epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, criterion, config, args.start_epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config, PAD_IDX, loss_scaler, TD_train_dict)
        lr_scheduler.step(epoch)
        TD_train_dict['lr_scheduler'].step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'text_decoder': TD_train_dict['text_decoder'].state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)

        if min_loss > test_stats["loss"]:
            min_loss = test_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'text_decoder': TD_train_dict['text_decoder'].state_dict(),
                        'epoch': epoch,
                        # 'args': args,
                    }, checkpoint_path)
        
        print(f"* DEV loss {test_stats['loss']:.3f} Min DEV loss {min_loss}")
        if args.run:
            args.run.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'training/masked_lm_loss':train_stats['masked_lm_loss'], 'dev/dev_loss':test_stats['loss'], 'dev/min_loss': min_loss})


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        # torch.distributed.barrier()#确保所有进程都在此行同步
        checkpoint = torch.load(args.output_dir+'/best_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, PAD_IDX, loss_scaler, TD_train_dict, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)

    for step, (src_input, tgt_input, masked_tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text, ground_truth = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)

            # margin = max(10, int((frames_feature.shape[1] // tgt_input['input_ids'].shape[1] + 1) * 2.3)) /2
            # num_negative = 10
            # margin = min(margin, int((frames_feature.shape[1] - num_negative) / 2)) #ensure num_frames margin for negative sampling
            # cl_loss = cl_criterion(frames_feature, margin=margin)

            total_loss = (loss_imgs + loss_texts)/2.
            # total_loss1 = total_loss + 0.1 * M_loss
        loss_scaler(total_loss, optimizer)

        # update the text decoder parames
        if step % 5 == 0:
            TD_train_dict['optimizer'].zero_grad()
            with torch.cuda.amp.autocast():
                # lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model.module.model_txt)
                #取消分布式训练后显示没有module,尝试以下代码
                lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model.model_txt)
                #尝试将mlm建模换成翻译模块
                # lm_logits = TD_train_dict['text_decoder'](tgt_input, src_input, model.model_images)
                masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].cuda().view(-1)) * args.loss_lambda
            loss_scaler(masked_lm_loss, TD_train_dict['optimizer'])

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        # metric_logger.update(cl_loss=cl_loss.item())
        # metric_logger.update(M_loss=M_loss.item())
        metric_logger.update(masked_lm_loss=masked_lm_loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(td_lr=TD_train_dict['optimizer'].param_groups[0]["lr"])

        if (step+1) % 10 == 0 and utils.is_main_process():
            visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
            # utils.visualization([visual_map,])

    if args.run:
        args.run.log({'epoch':epoch+1,'epoch/train_loss':loss_value, 'epoch/masked_lm_loss':masked_lm_loss.item()})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)

    with torch.no_grad():
        for step, (src_input, tgt_input, masked_tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, print_freq, header)):

            logits_per_image, logits_per_text, ground_truth = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)

            lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model_without_ddp.model_txt)
            #尝试将mlm建模换成翻译模块
            # lm_logits = TD_train_dict['text_decoder'](tgt_input, src_input, model_without_ddp.model_images)

            masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].cuda().view(-1))
            total_loss = (loss_imgs + loss_texts)/2.

            metric_logger.update(loss=total_loss.item())
            # metric_logger.update(M_loss=M_loss.item())
            metric_logger.update(masked_lm_loss=masked_lm_loss.item())

            if (step+1) % 10 == 0 and utils.is_main_process():
                visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
                # utils.visualization([visual_map, ])

    if args.run:
        args.run.log({'epoch':epoch+1,'epoch/dev_loss':total_loss.item()})
    
    metric_logger.synchronize_between_processes()
    print("* Averaged stats:", metric_logger)
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def setup_run(args, config):
    if args.log_all:
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("training/*", step_metric="epoch")
        run.define_metric("dev/*", step_metric="epoch")
    else:
        if utils.is_main_process():
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            run.name = args.output_dir.split('/')[-1]
        else:
            os.environ["WANDB_MODE"] = 'disabled'
            run = False

    return run
if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    # wandb.init a run if logging, otherwise return None
    args.run = setup_run(args, config)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)