# *torch
from pickletools import optimize #用于优化pickle文件的大小
# from sched import scheduler
import torch
import torch.nn as nn #提供了构建神经网络所需的各种层和函数
import torch.backends.cudnn as cudnn #提供GPU加速的支持
from torch.optim import lr_scheduler as scheduler #PyTorch的优化器库，提供了学习率调度器，用于动态调整学习率
from torch.utils.data import DataLoader #PyTorch的数据加载器，用于方便地加载和处理数据。
from torch.nn.utils.rnn import pad_sequence #PyTorch的RNN实用工具，提供了序列填充的功能，用于处理不同长度的序列数据。


# *transformers，huggingface的预训练模型库，可以直接使用预训练好的模型
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig 
#Hugging Face Transformers库中的MBart模型，用于条件生成任务，如翻译和摘要。MBart tokenizer用于将文本数据转换为模型可接受的输入格式。MBart配置，用于定义模型的超参数和配置选项。
from transformers.models.mbart.modeling_mbart import shift_tokens_right
#shift_tokens_right用于将输入的文本向右移动k个位置，以便在模型中正确填充输入。

# *user-defined
from models import gloss_free_model,Text_Decoder
from datasets import S2T_Dataset
import utils as utils

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
import test as test
import wandb
import copy
from pathlib import Path
from typing import Iterable, Optional
import math, sys
from loguru import logger

from hpman.m import _
import hpargparse

# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER
try:
    from nlgeval import compute_metrics
except:
    print('Please install nlgeval package.')


# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import NativeScaler
from timm.optim import AdamW

# global definition
from definition import *
import torch.nn.functional as F
import pickle  
import sacrebleu1,Rouge

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


def bleu11(references, hypotheses, level='word'):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    if level=='char':
        #split word
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(r)) for r in hypotheses]
    bleu_scores = sacrebleu1.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores

def rouge(references, hypotheses, level='word'):
    if level=='char':
        hyp = [list(x) for x in hypotheses]
        ref = [list(x) for x in references]
    else:
        hyp = [x.split() for x in hypotheses]
        ref = [x.split() for x in references]
    a = Rouge.rouge([' '.join(x) for x in hyp], [' '.join(x) for x in ref])
    return a['rouge_l/f_score']*100


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
        elif self.adaptive_method=="exp134":
            loss_weight = [torch.exp(-0.5 *  torch.pow(torch.tensor(item-60), torch.tensor(2))/torch.pow(torch.tensor(30.0), torch.tensor(2))) for item in weight]
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
    
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

class CrossEn12(nn.Module):
    """cross entroy loss"""
    def __init__(self):
        super(CrossEn11, self).__init__()
        self.sentence_avg = True
        self.eps = 0.2
        self.lm_eps = 0.2
        self.padding_idx=1
        self.token_scale = 0.1
        self.sentence_scale = 0.3
        self.pretrain_steps = 1
        self.lm_rate = 0.01
        self.finetune_fix_lm=True

    def forward(self,logits,logits1,label, reduce=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
    
        nmt_loss, nmt_nll_loss, lm_loss, lm_nll_loss = self.compute_loss(logits,logits1,label, reduce=False)
        # sample_size = (
        #     sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        # )
        # logging_output = {
        #     "loss": nmt_loss.data,
        #     "nll_loss": nmt_nll_loss.data,
        #     "lm_loss": lm_loss.data,
        #     "lm_nll_loss": lm_nll_loss.data,
        #     "ntokens": sample["ntokens"],
        #     "nsentences": sample["target"].size(0),
        #     "sample_size": sample_size,
        # }
        # for key in log:
        #     logging_output[key] = log[key]
        # if self.report_accuracy:
        #     n_correct, total = self.compute_accuracy(model, net_output, sample)
        #     logging_output["n_correct"] = utils.item(n_correct.data)
        #     logging_output["total"] = utils.item(total.data)
        if self.finetune_fix_lm and self.pretrain_steps==1:
            lm_loss = lm_loss.detach()

        loss = nmt_loss + self.lm_rate * lm_loss
        return loss

    def compute_loss(self, logits,logits1,label, reduce=False):
        # Rather than using ori api, rewrite cbmi calculation here
        # =========== CBMI-based adaptive loss =========== #
        nmt_logits = logits
        lm_logits = logits1
        nmt_probs = F.softmax(logits, dim=-1).reshape(-1, nmt_logits.shape[-1])
        lm_probs = F.softmax(logits1, dim=-1).reshape(-1, lm_logits.shape[-1])
        nmt_lprobs = torch.log(nmt_probs)
        lm_lprobs = torch.log(lm_probs)
        
        target = label
        pad_mask = target.ne(self.padding_idx)
        shape = target.shape
        target = target.reshape(-1)
        # if target.dim() == nmt_logits.dim():
        #     target = target.unsqueeze(-1)

        nmt_loss, nmt_nll_loss = label_smoothed_nll_loss(
            nmt_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=False,
        )
        lm_loss, lm_nll_loss = label_smoothed_nll_loss(
            lm_lprobs,
            target,
            self.lm_eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        if  self.pretrain_steps==1:
            cbmi = torch.log(nmt_probs / (lm_probs + 1e-9))    # in case that lm_probs are too little
            cbmi = cbmi.detach()
            golden_cbmi = torch.gather(cbmi, -1, index=target.unsqueeze(-1))
            # token-weight
            token_cbmi = golden_cbmi.reshape(shape)
            mean_token_cbmi = (token_cbmi * pad_mask).sum(-1, keepdims=True) / pad_mask.sum(-1, keepdims=True)
            std_token_cbmi = torch.sqrt(torch.sum((token_cbmi - mean_token_cbmi) ** 2 * pad_mask, -1, keepdims=True) / pad_mask.shape[-1])
            norm_token_cbmi = (token_cbmi - mean_token_cbmi) / std_token_cbmi
            token_weight = torch.where(self.token_scale * norm_token_cbmi + 1.0 >= 0, 
                                       self.token_scale * norm_token_cbmi + 1.0, 
                                       torch.zeros_like(norm_token_cbmi))
            # sentence-weight
            sentence_cbmi = mean_token_cbmi
            mean_sentence_cbmi = sentence_cbmi.mean(0, keepdims=True)
            std_sentence_cbmi = torch.sqrt(torch.sum((sentence_cbmi - mean_sentence_cbmi) ** 2, 0, keepdims=True) / pad_mask.shape[-1])
            norm_sentence_cbmi = (sentence_cbmi - mean_sentence_cbmi) / std_sentence_cbmi
            sentence_weight = torch.where(self.sentence_scale * norm_sentence_cbmi + 1.0 >= 0, 
                                          self.sentence_scale * norm_sentence_cbmi + 1.0, 
                                          torch.zeros_like(norm_sentence_cbmi))
            # final-weight
            weight = token_weight * sentence_weight
            weight = weight.detach()
            nmt_loss = nmt_loss.reshape(shape)
            nmt_loss = weight * nmt_loss 
            # logging output
            mean_cbmi = (token_cbmi * pad_mask).sum() / pad_mask.sum()
            std_cbmi = torch.sqrt(((token_cbmi - mean_cbmi) ** 2 * pad_mask).sum() / pad_mask.sum())
            max_weight = weight.max()
            min_weight = weight.min()
            zero_rate = torch.div((weight.eq(0) * pad_mask).sum(), pad_mask.sum())
        else:
            mean_cbmi = 0.0
            std_cbmi = 0.0
            max_weight = 0.0
            min_weight = 0.0
            zero_rate = 0.0
            
        logging_output = {
            "mean_cbmi": mean_cbmi, 
            "std_cbmi": std_cbmi, 
            "max_weight": max_weight, 
            "min_weight": min_weight, 
            "zero_rate": zero_rate,
        }
        if reduce:
            nmt_loss = nmt_loss.sum()
            nmt_nll_loss = nmt_nll_loss.sum()
        else:
            nmt_loss = nmt_loss.mean()
            nmt_nll_loss = nmt_nll_loss.mean()

        return nmt_loss, nmt_nll_loss, lm_loss, lm_nll_loss
    
def get_args_parser():
    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # * distributed training parameters
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
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
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
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free.yaml')

    # *Drop out params
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # * Mixup params，用于控制图像混合操作
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')#用于设置混合操作的α值。如果该值大于0，将启用混合操作。默认值为0.8
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)') #用于设置切割混合操作的α值。如果该值大于0，将启用切割混合操作。默认值为1.0
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')#于设置切割混合操作的最小和最大比例。如果设置了这个参数，将启用切割混合操作，并覆盖α值。
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled') #在混合操作和切割混合操作都启用时，执行混合操作或切割混合操作的概率。默认值为1.0。
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled') #混合操作和切割混合操作都启用时，切换到切割混合操作的概率。
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')#用于设置混合操作和切割混合操作的应用方式。它可以是"batch"、"pair"或"elem"之一。
    
    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    
    # * visualization
    parser.add_argument('--visualize', action='store_true')

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

    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['transformer'])


    train_data = S2T_Dataset(path=config['data']['train_label_path'],path1=config['data']['train_head_rgb_input'], tokenizer = tokenizer, config=config, args=args, phase='train',training_refurbish=False)
    print(train_data)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem)
    
    
    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], path1=config['data']['dev_head_rgb_input'],tokenizer = tokenizer, config=config, args=args, phase='val',training_refurbish=False)
    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_sampler = torch.utils.data.RandomSampler(dev_data) 

    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)
    
    test_data = S2T_Dataset(path=config['data']['test_label_path'],path1=config['data']['test_head_rgb_input'], tokenizer = tokenizer, config=config, args=args, phase='test',training_refurbish=False)
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.RandomSampler(test_data)

    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)
    

    print(f"Creating model:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['transformer'], src_lang = 'de_DE', tgt_lang = 'de_DE') # MBartTokenizer.from_pretrained('facebook/mbart-large-50')
    #MBartTokenizer从预训练的transformer模型中加载分词器。我们传递config['model']['transformer']作为参数，以指定要使用的预训练模型。src_lang和tgt_lang参数分别指定源语言和目标语言。
    model = gloss_free_model(config, args)
    model.to(device)
    print(model)

    #正式训练中的解码器    
    # text_decoder = Text_Decoder(config).to(device) #创建一个文本解码器模型 text_decoder，并将其转移到指定的设备上。
    # if args.distributed: #将 text_decoder 包装在 DistributedDataParallel 中，以便在多个GPU上进行分布式训练。
    #     text_decoder = torch.nn.parallel.DistributedDataParallel(text_decoder, device_ids=[args.gpu], find_unused_parameters=True)
    # # optimizer_td = AdamW(text_decoder.module.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98)) 
    # #取消分布式训练后显示没有module,尝试以下代码
    # optimizer_td = AdamW(text_decoder.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98)) #用于优化文本解码器的参数，设置了学习率和权重衰减等参数
    # lr_scheduler_td = scheduler.CosineAnnealingLR(
    #             optimizer=optimizer_td,
    #             eta_min=1e-8,
    #             T_max=args.epochs,
    #         ) #创建一个学习率调度器 lr_scheduler_td，使用余弦退火策略，初始化时学习率为 1e-3，最小学习率为 1e-8，并设置最大周期为 args.epochs。
    # TD_train_dict = dict(
    #     optimizer = optimizer_td,
    #     lr_scheduler = lr_scheduler_td,
    #     text_decoder = text_decoder
    # ) #将优化器、学习率调度器和文本解码器打包成一个字典 TD_train_dict，以便在训练过程中使用


    if args.finetune: #如果为True，表示我们需要加载预训练的模型参数来进行微调。
        print('***********************************')
        print('Load parameters for Visual Encoder...')
        print('***********************************')
        
        # Function to check if a tensor can have gradients
        def can_require_grad(tensor):
            return torch.is_tensor(tensor) and tensor.dtype in (
                torch.float32, torch.float64, torch.complex64, torch.complex128)
        state_dict = torch.load(args.finetune, map_location='cpu') #torch.load加载预训练的模型参数。map_location='cpu'参数表示我们将模型参数加载到CPU上。

        new_state_dict = OrderedDict() #创建一个新的有序字典new_state_dict，用于存储需要微调的模型参数
        for k, v in state_dict['model'].items(): #遍历预训练的模型参数，将需要微调的参数添加到new_state_dict中。我们只微调卷积层和transformer编码器层。
            if 'conv_2d' in k or 'conv_1d' in k:
                k = 'backbone.'+'.'.join(k.split('.')[2:])
                new_state_dict[k] = v
            if 'trans_encoder' in k:
                k = 'mbart.model.encoder.'+'.'.join(k.split('.')[2:])
                new_state_dict[k] = v
            #加载visual_head参数
            if 'visual_head' in k:
                k='visual_head.'+'.'.join(k.split('.')[2:])
                new_state_dict[k] = v
            if "mapping" in k:
                k='mapping.'+'.'.join(k.split('.')[2:])
                new_state_dict[k] = v
                
            # if 'decoder' in k:
            #     k = 'mbart.model.decoder.'+'.'.join(k.split('.')[1:])
            #     if can_require_grad(v):
            #         v.requires_grad = True
            #     new_state_dict[k] = v
            # if 'lm_head.weight' in k:
            #     k = 'mbart.'+'.'.join(k.split('.')[0:])
            #     new_state_dict[k] = v
            # if "localattention" in k:
            #     k='localattention.'+'.'.join(k.split('.')[2:])
            #     new_state_dict[k] = v
            # if "layer_norm" in k:
            #     k='layer_norm.'+'.'.join(k.split('.')[2:])
            #     new_state_dict[k] = v
            
            if 'text_decoder' in state_dict and config['model']["decoder_type"] != "LLMD": #如果text_decoder在预训练的模型参数中，我们也将需要微调的解码器层参数添加到new_state_dict中
                for k, v in state_dict['text_decoder'].items():
                    if 'decoder' in k:
                        k = 'mbart.model.decoder.'+'.'.join(k.split('.')[2:])
                        if can_require_grad(v):
                            v.requires_grad = True
                        new_state_dict[k] = v
                    
                    # if 'lm_head.weight' in k:
                    #     k = 'mbart.'+'.'.join(k.split('.')[0:])
                    #     new_state_dict[k] = v
                    # if 'final_logits_bias' in k:
                    #     k = 'mbart.'+'.'.join(k.split('.')[0:])
                    #     new_state_dict[k] = v         

        #从别处加载的mbart_decoder参数
        # decoder_dict=torch.load("/home/zfc2b/CV-SLT-main/experiments/configs/SingleStream/phoenix-2014t_g2t/ckpts/best.ckpt", map_location='cpu')
        # print(decoder_dict['translation_network']['model']['model']['decoder']['embed_tokens']['weight'].shape)
        # for k, v in decoder_dict.items():
        #     if 'translation_network.model.model.decoder.embed_tokens.weight' in k:
        #         k='mbart.'+'.'.join(k.split('.')[2:])
        #         new_state_dict[k] = v
        #     if 'translation_network.model.model.decoder.embed_positions.weight' in k:
        #         k='mbart.'+'.'.join(k.split('.')[2:])
        #         new_state_dict[k] = v
        #     if "model.model.decoder.layers.0" in k:
        #         k='mbart.model.decoder.layers.0.'+'.'.join(k.split('.')[6:])
        #         new_state_dict[k] = v
        #     if "model.model.decoder.layers.1" in k:
        #         k="mbart.model.decoder.layers.1."+'.'.join(k.split('.')[6:])
        #         new_state_dict[k] = v
        #     if "model.model.decoder.layers.2" in k:
        #         k='mbart.model.decoder.layers.2.'+'.'.join(k.split('.')[6:])
        #         new_state_dict[k] = v
        #     if 'model.final_logits_bias' in k:
        #         k='mbart.'+'.'.join(k.split('.')[2:])
        #         new_state_dict[k] = v
            # if 'model.model.shared.weight' in k:
            #     k='mbart.'+'.'.join(k.split('.')[2:])
            #     new_state_dict[k] = v
            # if 'translation_network.model.model.decoder.layernorm_embedding.weight' in k:
            #     k='mbart.'+'.'.join(k.split('.')[2:])
            #     new_state_dict[k] = v
            # if 'translation_network.model.model.decoder.layernorm_embedding.bias' in k:
            #     k='mbart.'+'.'.join(k.split('.')[2:])
            #     new_state_dict[k] = v
            # if 'translation_network.model.model.decoder.layer_norm.weight' in k:
            #     k='mbart.'+'.'.join(k.split('.')[2:])
            #     new_state_dict[k] = v
            # if 'translation_network.model.model.decoder.layer_norm.bias' in k:
            #     k='mbart.'+'.'.join(k.split('.')[2:])
            #     new_state_dict[k] = v
            # if 'translation_network.model.lm_head.weight' in k:
            #     k='mbart.'+'.'.join(k.split('.')[2:])
            #     new_state_dict[k] = v


        # *replace the word embedding
        model_dict = torch.load(config['model']['transformer']+'/pytorch_model.bin', map_location='cpu') #加载预训练的transformer模型的参数
        for k, v in model_dict.items(): #遍历预训练的模型参数，将需要微调的参数添加到new_state_dict中。我们只微调词嵌入层和位置嵌入层
            if 'decoder.embed_tokens.weight' in k:
                k = 'mbart.' + k
                new_state_dict[k] = v
            if 'decoder.embed_positions.weight' in k:
                k = 'mbart.' + k
                new_state_dict[k] = v

        ret = model.load_state_dict(new_state_dict, strict=False) #使用model.load_state_dict方法将new_state_dict中的参数加载到模型中。strict=False参数表示我们不严格检查模型的参数和new_state_dict中的参数是否完全匹配
        print('Missing keys: \n', '\n'.join(ret.missing_keys)) #打印缺失的参数
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys)) #打印多余的参数

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    # 选一个代表性 batch（优先 dev，再退回 train）
    try:
        example_batch = next(iter(dev_dataloader))
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

    lr_scheduler = scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=1e-8,
                T_max=args.epochs,
            ) #创建一个CosineAnnealingLR的实例，用于调整学习率。eta_min参数表示最小学习率，T_max参数表示最大迭代轮数。
    loss_scaler = NativeScaler() #创建一个NativeScaler对象。NativeScaler用于在训练过程中对损失进行缩放，以防止数值溢出

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active: #根据命令行参数决定是否使用数据增强技术，以及选择合适的损失函数。
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=0.2, num_classes=2454) #创建一个Mixup实例。Mixup支持混合、剪切混合和混合最小最大。

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy() #SoftTargetCrossEntropy用于处理混合标签。
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)
        criterion1 =CrossEn11('word_freq_dict.pkl')
        criterion2 =CrossEn11('word_freq_dict.pkl')
    
    output_dir = Path(args.output_dir)
    if args.resume: #如果我们需要从已保存的模型参数中恢复训练，则从args.resume参数中读取模型参数。
        print('Resuming Model Parameters... ')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True) #用于加载检查点中的模型参数到当前的模型中，checkpoint['model']是检查点中的模型参数部分。strict=True表示只有当模型结构和检查点中的结构完全一致时，才加载参数
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch']+1

    if args.eval:
        if not args.resume:#如果用户在评估模式下没有提供--resume参数，那么会打印出一个警告信息，提示用户需要指定一个训练过的模型
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        epoch1=300
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, criterion1,epoch1,config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f} ")
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion,criterion1, epoch1,config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}") #打印模型在测试集上的BELU-4分数。

        return
 
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch) #train_sampler的epoch属性，以便在训练过程中对数据进行随机采样
        
        train_stats = train_one_epoch(args, model, criterion,criterion1, train_dataloader, optimizer, device, epoch, config, loss_scaler, mixup_fn)
        lr_scheduler.step(epoch) #学习率调度器会根据当前的周期和训练统计信息动态地调整学习率。

        if args.output_dir and utils.is_main_process(): #是否提供了--output_dir参数，并且当前进程是主进程
            checkpoint_paths = [output_dir / f'checkpoint.pth'] #设置了检查点文件的路径
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({ #用于保存模型的检查点。它包括模型的状态字典、优化器的状态字典、学习率调度器的状态字典和当前周期数。这些信息会被保存到指定的检查点文件中
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)
        
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, criterion1,epoch,config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}")

        if max_accuracy < test_stats["belu4"]:
            max_accuracy = test_stats["belu4"]
            if args.output_dir and utils.is_main_process():
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max BELU-4: {max_accuracy:.2f}%')
        if utils.is_main_process():
            wandb.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'dev/dev_loss':test_stats['loss'], 'dev/Bleu_4':test_stats['belu4'], 'dev/Best_Bleu_4': max_accuracy})
            #用于将训练和测试的统计信息上传到Wandb。Wandb是一个用于跟踪、可视化和分享机器学习实验结果的平台
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters} #用于记录训练和测试的统计信息，例如损失值、BLEU分数、模型参数数量等
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f: #文件会被以追加模式打开，这样新的统计信息会被追加到文件的末尾。
                f.write(json.dumps(log_stats) + "\n") #用于将log_stats字典转换为JSON格式的字符串，并将其写入文件。每行一个统计信息。
        # gc.collect()#执行垃圾回收
        # if hasattr(torch.cuda, 'empty_cache'):#如果torch.cuda具有'empty_cache'属性，清空CUDA缓存。
        #     torch.cuda.empty_cache()
    # Last epoch
    test_on_last_epoch = True #是否在最后一个周期上执行测试
    if test_on_last_epoch and args.output_dir:
        checkpoint = torch.load(args.output_dir+'/best_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, criterion1,epoch,config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}")
        
        test1_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion, criterion1,epoch,config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test1_stats['belu4']:.2f}")

        #准备日志统计数据
        log_stats = {**{f'train_{k}': v for k, v in test_stats.items()},
                     **{f'test_{k}': v for k, v in test1_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f: #文件会被以追加模式打开，这样新的统计信息会被追加到文件的末尾。
                f.write(json.dumps(log_stats) + "\n") #用于将log_stats字典转换为JSON格式的字符串，并将其写入文件。每行一个统计信息。

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time))) #用于将总时间转换为易读的字符串格式。
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,criterion1:torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, loss_scaler, mixup_fn=None, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode) #将模型设置为训练模式，启用dropout和batch normalization等训练时需要的操作。

    metric_logger = utils.MetricLogger(delimiter="  ") #用于记录训练过程中的各种指标的日志记录器，使用了指定的分隔符。
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))#向日志记录器中添加了一个平滑的学习率指标，用于记录学习率的变化，窗口大小为1，输出格式为6位小数。
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #用于遍历训练数据的batch。在每次迭代中，它会获取一个包含源语言输入和目标语言输入的tuple
        total_loss = 0.0

        out_logits, output = model(src_input, tgt_input) #计算模型对输入的预测输出。返回解码器的输出 logits 和编码器的最后一个隐藏状态
        label = tgt_input['input_ids'].reshape(-1) #获取目标语言输入的label，将目标语言输入的'input_ids' reshape为1D张量，用于计算损失
        logits = out_logits.reshape(-1,out_logits.shape[-1]) #将logits reshape为2D张量，以便与label一起计算损失
        # tgt_loss = criterion(logits, label.to(device, non_blocking=True)) #计算模型输出和目标语言输入之间的损失。
        tgt_loss = criterion1(out_logits, label.to(device, non_blocking=True))
        #采用改进的损失函数
        # if epoch>200:
        #     tgt_loss = criterion1(out_logits, label.to(device, non_blocking=True))
        # else:
        #     tgt_loss = criterion(logits, label.to(device, non_blocking=True)) #计算模型输出和目标语言输入之间的损失。
        total_loss += tgt_loss

        optimizer.zero_grad() #将优化器参数的梯度归零
        total_loss.backward() #计算损失的梯度
        optimizer.step() #使用优化器更新参数

        loss_value = total_loss.item() #获取损失值
        if not math.isfinite(loss_value): #检查损失值是否为有限数，如果不是，则停止训练。
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value) #更新日志记录器中的损失值
        metric_logger.update(lr=optimizer.param_groups[0]["lr"]) #更新日志记录器中的学习率
        metric_logger.update(lr_mbart=round(float(optimizer.param_groups[1]["lr"]), 8)) #更新日志记录器中的MBart学习率值。

        if (step+1) % 10 == 0 and args.visualize and utils.is_main_process():
            utils.visualization(model.module.visualize()) #执行模型的可视化操作
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes() #用于在所有进程之间同步统计信息，确保所有进程都包含了最新的统计数据。
    print("Averaged stats:", metric_logger) #打印平均的统计信息。
 
    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()} #这行代码返回一个字典，其中包含了所有统计信息的平均值

def evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion,criterion1, epoch, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device):
    model.eval() #设置为评估模式，在这个模式下，模型的dropout和batch normalization层会被禁用，以确保模型的预测结果是确定性的。

    metric_logger = utils.MetricLogger(delimiter="  ") #创建一个MetricLogger对象，用于记录评估过程中的各种统计信息
    header = 'Test:'

    with torch.no_grad(): #这行代码开启一个不需要梯度计算的上下文，因为在评估阶段不需要计算梯度
        tgt_pres = []
        tgt_refs = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, 10, header)):

            out_logits, output = model(src_input, tgt_input)
            total_loss = 0.0
            label = tgt_input['input_ids'].reshape(-1)
            
            logits = out_logits.reshape(-1,out_logits.shape[-1])
            # tgt_loss = criterion(logits, label.to(device))
            tgt_loss = criterion1(out_logits, label.to(device))
            #改进的损失函数
            # if epoch>200:
            #     tgt_loss = criterion1(out_logits, label.to(device))
            # else:
            #     tgt_loss = criterion(logits, label.to(device))
            total_loss += tgt_loss

            metric_logger.update(loss=total_loss.item())#更新日志记录器中的损失值
            
            output = model_without_ddp.generate(src_input, max_new_tokens=150, num_beams = 4,
                        decoder_start_token_id=tokenizer.lang_code_to_id['de_DE']
                        ) #使用模型生成目标语言的序列。model_without_ddp是去掉了分布式并行（DDP）的包装后的模型。generate方法使用了num_beams=4的束搜索解码和最大新令牌数150的限制

            tgt_input['input_ids'] = tgt_input['input_ids'].to(device) #将tgt_input的输入id移到GPU上
            for i in range(len(output)):#将生成的输出id转换为相应的文本序列
                tgt_pres.append(output[i,:])
                tgt_refs.append(tgt_input['input_ids'][i,:])
            
            if (step+1) % 10 == 0 and args.visualize and utils.is_main_process():
                utils.visualization(model_without_ddp.visualize()) #可视化操作

    pad_tensor = torch.ones(200-len(tgt_pres[0])).to(device) #将生成的序列补全到最大长度200，建一个全1的张量，长度为200减去第一个预测序列的长度，然后将其移动到正确的设备（CPU或GPU）
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0) #将预测序列的第一个序列和全1张量连接在一起，通过dim=0指定沿着维度0（即序列的长度）进行连接。
    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=PAD_IDX) #将预测序列进行填充，填充值为PAD_IDX，并将它们转换为一个批次的第一个维度

    pad_tensor = torch.ones(200-len(tgt_refs[0])).to(device)
    tgt_refs[0] = torch.cat((tgt_refs[0],pad_tensor.long()),dim = 0)
    tgt_refs = pad_sequence(tgt_refs,batch_first=True,padding_value=PAD_IDX)

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True) #将预测序列转换为对应的文本序列
    tgt_refs = tokenizer.batch_decode(tgt_refs, skip_special_tokens=True)#将目标序列转换为对应的文本序列
    #用tokenizer对象的batch_decode方法将填充或截断后的序列从索引值转换为实际的文本。skip_special_tokens=True表示在解码过程中跳过特殊的标记，如[CLS]和[SEP]。
    bleu = BLEU() #初始化一个BLEU对象，用于计算BLEU得分
    bleu_s = bleu.corpus_score(tgt_pres, [tgt_refs]).score #计算BLEU得分
    # metrics_dict['belu4']=bleu_s

    metric_logger.meters['belu4'].update(bleu_s) #更新belu4的Meter对象
    

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* BELU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.belu4, losses=metric_logger.loss)) #输出BLEU-4分数和训练过程中的平均损失
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir+'/tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i]+'\n') #将生成的序列保存到tmp_pres.txt文件中
        with open(args.output_dir+'/tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i]+'\n') #将目标序列保存到tmp_refs.txt文件中
        print('\n'+'*'*80)
        # metrics_dict = compute_metrics(hypothesis=args.output_dir+'/tmp_pres.txt',
        #                    references=[args.output_dir+'/tmp_refs.txt'],no_skipthoughts=True,no_glove=True)#计算BLEU、METEOR、ROUGE-L、CIDEr等指标，no_skipthoughts=True和no_glove=True表示不使用Skip-thoughts和GloVe模型。

        #cvslt弄过来的指标计算函数
        bleu_dict = bleu11(references=tgt_refs, hypotheses=tgt_pres, level="word")#计算 BLEU 值。
        rouge_score = rouge(references=tgt_refs, hypotheses=tgt_pres, level="word")#计算 ROUGE 值。
        print(", ".join('{}: {:.2f}'.format(k, v) for k, v in bleu_dict.items()))#记录 BLEU 值。
        print('ROUGE: {:.2f}'.format(rouge_score))
        print('*'*80)
        #准备日志统计数据
        log_stats1 = {**{'{}'.format(k): '{:.2f}'.format(v) for k, v in bleu_dict.items()},
                     'ROUGE:': '{:.2f}'.format(rouge_score)}
        if args.output_dir:
            with (Path(args.output_dir) /"log.txt").open("a") as f: #文件会被以追加模式打开，这样新的统计信息会被追加到文件的末尾。
                f.write(json.dumps(log_stats1) + "\n") #用于将log_stats字典转换为JSON格式的字符串，并将其写入文件。每行一个统计信息。
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} #返回一个字典，其中包含了metric_logger中记录的所有指标的平均值。

if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled' #设置了Wandb的环境变量。Wandb是一个用于跟踪和可视化深度学习训练过程的工具。
    if utils.is_main_process(): #是否在主进程中，在分布式训练中，通常只有一个主进程会执行Wandb的初始化和其他一些操作
        wandb.init(project='GF-SLT',config=config) #主进程中初始化Wandb，并指定了项目名称和配置文件
        wandb.run.name = args.output_dir.split('/')[-1] #设置当前Wandb的运行名称，即输出文件夹的名字
        wandb.define_metric("epoch") #定义了一个名为"epoch"的度量。这是一个在训练过程中通常会记录的度量
        wandb.define_metric("training/*", step_metric="epoch") #定义了一个名为"training/*"的度量，其中"*"表示可以有任意数量的子度量。这个度量将在每个训练周期结束时记录
        wandb.define_metric("dev/*", step_metric="epoch")  #定义了一个名为"dev/*"的度量，其中"*"表示可以有任意数量的子度量。这个度量将在每个验证周期结束时记录
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    gc.collect()#执行垃圾回收
    if hasattr(torch.cuda, 'empty_cache'):#如果torch.cuda具有'empty_cache'属性，清空CUDA缓存。
        torch.cuda.empty_cache()
    main(args, config)

