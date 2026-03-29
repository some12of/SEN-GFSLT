from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask

import torchvision
from torch.nn.utils.rnn import pad_sequence
#import pytorchvideo.models.x3d as x3d
import utils as utils

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding, MBartEncoderLayer, _expand_mask

from collections import OrderedDict


import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *

from hpman.m import _
from pathlib import Path
#visual_head
from Visualhead import VisualHead
from Tokenizer import GlossTokenizer_S2G
from PDE import DisTrans
import objectives
from signcl import SignCL
from module_cross import Transformer as TransformerClip
from cluster import CTM, TCBlock
from transformers import RobertaConfig
from bert_model import BertCrossLayer
from transformer_layers import DeformableMultiHeadedAttention,PositionwiseFeedForward
from utils1 import  MaskedNorm
from confidence import GraphEmbt,GraphEmbv,EncoderSimilarity

class PositionalEncoding(nn.Module):#模型中的序列数据添加位置信息
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):#序列最大长度默认5000
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)#计算每个元素的指数值，并将负号添加到前面，得到分子（denominator）
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)#创建一个表示位置的张量，调整形状为列向量
        pos_embedding = torch.zeros((maxlen, emb_size))#创建一个空的张量，用于存储位置编码
        pos_embedding[:, 0::2] = torch.sin(pos * den)#计算正弦值
        pos_embedding[:, 1::2] = torch.cos(pos * den)#计算余弦值
        pos_embedding = pos_embedding.unsqueeze(-2)#增加一个维度，使其成为3D张量形状调整为 (maxlen, emb_size, 1），以便与输入数据一起使用

        self.dropout = nn.Dropout(dropout)#实例化一个Dropout层，用于在训练过程中随机删除一些元素
        self.register_buffer('pos_embedding', pos_embedding)#注册一个缓冲区，该缓冲区包含位置编码

    def forward(self, token_embedding: Tensor):#接受token_embedding 参数（形状为 (batch_size, emb_size)），位置编码添加到 token_embedding 中，最后应用丢弃操作
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def make_resnet(name='resnet18'):#创建ResNet模型，接受一个参数（name），表示要使用的ResNet模型类型
    if name == 'resnet18':#创建一个预训练的 ResNet-18 模型
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))

    inchannel = model.fc.in_features #获取模型的全连接层前的通道数（即输入特征的维度)
    model.fc = nn.Identity() #删除全连接层，以便将其替换为新的线性层，全连接层替换为一个恒等函数（nn.Identity()），以保留输入特征不变
    return model

class resnet(nn.Module): #调用预训练的 ResNet 模型进行特征提取，并将每个序列的输出存储在一个批次张量中。这个类可以用于处理变长序列
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = [] # 创建一个列表，用于存储每个序列的输出
        start = 0 # 设置起始索引
        for length in lengths: #遍历长度列表
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True) #将 x_batch 列表转换为一个批次的张量，其中填充值为 PAD_IDX，并且以批次优先的方式存储
        return x
  
class TemporalConv(nn.Module): #对输入序列进行一维卷积操作，可以根据 conv_type 的值选择不同的卷积核大小
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0: #设置 kernel_size 属性，表示卷积核的大小
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"] #使用大小为 5 的卷积核，并且每隔 2 个元素进行一次池化操作
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = [] # 创建一个模块列表，用于存储一维卷积层和批量归一化层
        for layer_idx, ks in enumerate(self.kernel_size): #遍历 kernel_size 列表
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size #根据 layer_idx 动态设置输入通道数
            if ks[0] == 'P':#如果 ks[0] 等于 'P'，则添加一个最大池化层
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K': #如果 ks[0] 等于 'K'，则添加一个卷积层，并且添加一个批量归一化层和一个 ReLU 激活层。
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules) #使用 nn.Sequential 将 modules 列表中的层串联起来，作为 temporal_conv 属性。
    
    def forward(self, x): #接受一个输入张量 x，维度通常是 [batch_size, num_channels, length]，并返回经过一维卷积操作后的输出张量
        x = self.temporal_conv(x.permute(0,2,1)) #输入张量的维度重新排列，将其变为 [batch_size, length, num_channels]
        return x.permute(0,2,1)#经过 temporal_conv 处理后的输出数据重新排列，将其变为 [batch_size, num_channels, length]。这是为了恢复原始的输入顺序

class text_temporalConv(nn.Module): #对输入序列进行一维卷积操作，可以根据 conv_type 的值选择不同的卷积核大小
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(text_temporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        self.conv1=nn.Conv1d(self.input_size, self.hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv2=nn.Conv1d(self.input_size, self.hidden_size, kernel_size=2, stride=1, padding='same')
        self.conv3=nn.Conv1d(self.input_size, self.hidden_size, kernel_size=3, stride=1, padding='same')

        # Sentence feature extraction using average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feed_forward = PositionwiseFeedForward(
            input_size=1024, ff_size=2048, dropout=0.1
        )
        self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='batch')
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
        # if self.conv_type == 0: #设置 kernel_size 属性，表示卷积核的大小
        #     self.kernel_size = ['K3']
        # elif self.conv_type == 1:
        #     self.kernel_size = ['K5', "P2"] #使用大小为 5 的卷积核，并且每隔 2 个元素进行一次池化操作
        # elif self.conv_type == 2:
        #     self.kernel_size = ['K5', "P2", 'K5', "P2"]

        # modules = [] # 创建一个模块列表，用于存储一维卷积层和批量归一化层
        # for layer_idx, ks in enumerate(self.kernel_size): #遍历 kernel_size 列表
        #     input_sz = self.input_size if layer_idx == 0 else self.hidden_size #根据 layer_idx 动态设置输入通道数
        #     if ks[0] == 'P':#如果 ks[0] 等于 'P'，则添加一个最大池化层
        #         modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
        #     elif ks[0] == 'K': #如果 ks[0] 等于 'K'，则添加一个卷积层，并且添加一个批量归一化层和一个 ReLU 激活层。
        #         modules.append(
        #             nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
        #         )
        #         modules.append(nn.BatchNorm1d(self.hidden_size))
        #         modules.append(nn.ReLU(inplace=True))
        # self.temporal_conv = nn.Sequential(*modules) #使用 nn.Sequential 将 modules 列表中的层串联起来，作为 temporal_conv 属性。
    
    def forward(self, x,mask): #接受一个输入张量 x，维度通常是 [batch_size, num_channels, length]，并返回经过一维卷积操作后的输出张量
        x1=x.permute(0,2,1)
        x2 = self.conv1(x1)+ self.conv2(x1)+self.conv3(x1)
        x2=self.dropout1(x2.permute(0,2,1))
        x2=self.layer_norm(x2)
        x3=self.feed_forward(x2)+x1.permute(0,2,1)
        x3=self.bn1(x3, mask)
        x3= self.relu1(x3)
        x3=self.avg_pool(x3.permute(0,2,1)).squeeze(-1)
        return x3#经过 temporal_conv 处理后的输出数据重新排列，将其变为 [batch_size, num_channels, length]。这是为了恢复原始的输入顺序

def make_head(inplanes, planes, head_type): #用于根据 head_type 的值创建不同的全连接层。
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False) #线性全连接层，没有偏置
    else:
        return nn.Identity() #返回一个恒等层 nn.Identity()，表示不使用全连接层。

class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()

        self.model1=MBartForConditionalGeneration.from_pretrained(config['model']['transformer'])

        self.model_txt = self.model1.get_encoder() 
        # 使用预训练的 MBart 模型作为文本处理模型，只保留编码器部分
        self.lm_head = make_head(inplanes, planes, head_type) # 根据 head_type 的值创建不同的全连接层
        #文本特征细化模块
        # self.text_refine=text_temporalConv(1024,1024)

    def forward(self, tgt_input): #接受一个输入张量 tgt_input
        # text_embeds = self.model1.model.shared(tgt_input['input_ids'].cuda()) * (math.sqrt(self.model1.config.d_model))
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(), attention_mask=tgt_input['attention_mask'].cuda())[0] #将输入数据传递给 self.model_txt，并获取文本编码器的输出。
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]#创建一个张量，用于索引txt_logits，选取输入文本中每个单词的最大 logit 值作为输出
        # output1=output*0.9+self.text_refine(txt_logits,tgt_input['attention_mask'].cuda())*0.1
        # print(output.shape,output1.shape)
        return self.lm_head(output), txt_logits,txt_logits #返回经过全连接层处理后的输出和原始的 logit 输出


class ImageCLIP(nn.Module):
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear') :
        super(ImageCLIP, self).__init__()
        self.config = config #config 参数保存为类的属性。
        # self.model =  FeatureExtracter() #创建一个图像特征提取模型

        self.gloss_tokenizer = GlossTokenizer_S2G(config['model']['GlossTokenizer'])
        self.visual_head = VisualHead(cls_num=len(self.gloss_tokenizer))
        self.localattention=DeformableMultiHeadedAttention(
                query_type='not',
                query_nb=3,
                num_heads=8,
                size=512,
                dropout=0.5,
                num_keys=4,
            )
        self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        # self.feed_forward = PositionwiseFeedForward(
        #     input_size=512, ff_size=2048, dropout=0.5
        # )
        self.dropout = nn.Dropout(0.5)
        self.mapping = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=1024),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=1024, out_features=1024)
            )
        
        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_encoder()
        #使用预训练的 MBartForConditionalGeneration 模型作为图像处理模型，只保留编码器部分
        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes)) #创建一个可训练的参数，用于添加到图像特征向量的开头，作为分类头的输入

        self.lm_head = make_head(inplanes, planes, head_type) # 根据 head_type 的值创建不同的全连接层
        
    def forward(self, src_input): #src_input，该参数是一个包含图像特征的字典
       
        # x = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch']) # [b, n, c]，输入数据传递给 FeatureExtracter 模型
        # attention_mask = src_input['attention_mask']
        attention_mask = src_input['attention_mask']
        x=self.visual_head(src_input['input_ids'].cuda(),attention_mask.cuda())
        
        #local attention
        x_norm = self.layer_norm(x["gloss_feature"]).cuda()
        h= self.localattention(x_norm, x_norm, x_norm, attention_mask.cuda())
        h = self.dropout(h) + x["gloss_feature"]
        # h= self.feed_forward(h)
        # frame_features=h[:,:,:]
        x=self.mapping(h)
        frame_features=x[:,:,:]

        # frame_features=x["gloss_feature"][:,:,:]
        # x=self.mapping(x["gloss_feature"])

        # attention_mask1=attention_mask[:,:]
        #是一个掩码，用于指示输入序列中的哪些元素应该被考虑，哪些元素应该被忽略。通常使用 attention_mask 来忽略输入序列中的填充元素。
        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B) #self.cls_token是1, 1, inplanes)的张量，表示分类标记。将其重复 B 次，以匹配输入数据的批量大小。
        x = torch.cat((cls_token, x), dim=1) #将分类标记添加到图像特征向量的开头
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]
        #latten(1) 将 attention_mask 的形状从 (B, 64) 变为 (B, 64, 1)，然后 F.pad 在最后一个维度上添加了一个填充元素
        #(1, 0): 这是一个表示填充大小的元组。它表示在最后一个维度上添加一个填充元素，而在其他维度上不添加填充。

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        #对 x 进行编码，得到 outs，返回一个字典，其中包含编码器的输出和注意力分数
        last_hidden_state = outs['last_hidden_state']
        output = self.lm_head(last_hidden_state[:, 0, :])#对 last_hidden_state 中的第一个元素的进行线性变换
        #last_hidden_state 的形状是 (B, 65, inplanes)，其中 B 是批量大小，65 是序列长度，inplanes 是隐藏维度
        return output,last_hidden_state,attention_mask,frame_features

class Text_Decoder(nn.Module):#用于处理文本的解码器
    def __init__(self, config):
        super(Text_Decoder, self).__init__() 
        self.text_decoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_decoder()
        # 使用预训练的 MBartForConditionalGeneration 模型作为文本处理模型，只保留解码器部分
        self.lm_head = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_output_embeddings()
        # 使用预训练的 MBartForConditionalGeneration 模型作为文本处理模型，只保留输出嵌入
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).model.shared.num_embeddings)))
        # 注册一个缓冲区，用于存储文本解码器的最后一个线性层的偏置，final_logits_bias 是一个形状为 (1, num_embeddings) 的张量，初始值为 0

    def forward(self, tgt_input, masked_tgt_input, model_txt):#接受一个输入张量 tgt_input，masked_tgt_input，model_txt
        with torch.no_grad(): 
            _, encoder_hidden_states,_ = model_txt(masked_tgt_input)#使用模型_txt对 masked_tgt_input 进行编码，

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        #使用 shift_tokens_right 函数将 tgt_input['input_ids'] 向右移动一个位置，将输入文本的第一个元素替换为特殊符号，通常是一个特殊的标记，表示序列的开始，作为文本解码器的输入
        decoder_out = self.text_decoder(
                    input_ids = decoder_input_ids, # decoder_input_ids 的形状应为 (B, 64)，表示文本解码器的输入
                    attention_mask = tgt_input['attention_mask'].cuda(), # tgt_input['attention_mask'] 的形状应为 (B, 64)，指示解码器在生成输出时应该关注输入序列中的哪些部分
                    encoder_hidden_states = encoder_hidden_states, # encoder_hidden_states 的形状应为 (B, 65, 1024)，解码器会使用这些编码器隐藏状态来生成输出
                    encoder_attention_mask = masked_tgt_input['attention_mask'].cuda(),# encoder_attention_mask 的形状应为 (B, 65)，指示解码器在生成输出时应该关注编码器输出中的哪些部分
                    return_dict = True,
                    )

        # #尝试将mlm建模直接换成翻译模块
        # with torch.no_grad(): 
        #     _, encoder_hidden_states,attention_mask,_ = model_txt(masked_tgt_input)#使用模型_txt对 masked_tgt_input 进行编码，

        # decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        # #使用 shift_tokens_right 函数将 tgt_input['input_ids'] 向右移动一个位置，将输入文本的第一个元素替换为特殊符号，通常是一个特殊的标记，表示序列的开始，作为文本解码器的输入
        # decoder_out = self.text_decoder(
        #             input_ids = decoder_input_ids, # decoder_input_ids 的形状应为 (B, 64)，表示文本解码器的输入
        #             attention_mask = tgt_input['attention_mask'].cuda(), # tgt_input['attention_mask'] 的形状应为 (B, 64)，指示解码器在生成输出时应该关注输入序列中的哪些部分
        #             encoder_hidden_states = encoder_hidden_states[:,1:,:], # encoder_hidden_states 的形状应为 (B, 65, 1024)，解码器会使用这些编码器隐藏状态来生成输出
        #             encoder_attention_mask = attention_mask[:,1:].cuda(),# encoder_attention_mask 的形状应为 (B, 65)，指示解码器在生成输出时应该关注编码器输出中的哪些部分
        #             return_dict = True,
        #             )

        lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias #将 self.lm_head 计算得到的预测分数与 self.final_logits_bias 相加。self.final_logits_bias 是一个偏置项，通常用于调整模型的预测分数。

        return lm_logits
    
# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):#用于将注意力掩码从二维扩展到四维，以便与Transformer模型的其他张量（如inputs_embeds或hidden_states）进行广播。
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()#获取mask的形状(bsz, src_len)。
    tgt_len = tgt_len if tgt_len is not None else src_len #如果tgt_len未指定，则将其设置为src_len。

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype) #使用None和expand方法，将mask从[bsz, seq_len]扩展到[bsz, 1, tgt_seq_len, src_seq_len]。

    inverted_mask = 1.0 - expanded_mask #计算expanded_mask的取反数inverted_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min) #将inverted_mask中的1.0替换为torch.finfo(dtype).min指定数据类型的最小值，并将其他值保持不变。

def Wasserstein2(mu1, sigma1, mu2, sigma2): # 2W距离，传入图片和文本的均值和标准差
    bs1 = mu1.shape[0]
    bs2 = mu2.shape[0]
    mu1 = torch.stack([mu1]*bs2, dim=1)
    sigma1 = torch.stack([sigma1]*bs2, dim=1)
    mu2 = torch.stack([mu2]*bs1, dim=0)
    sigma2 = torch.stack([sigma2]*bs1, dim=0)
    p1 = torch.sum(torch.pow(mu1 - mu2, 2), dim=-1)
    p2 = torch.sum(torch.pow(sigma1 - sigma2, 2), dim=-1)
    return p1+p2, p1 
 
def gaussian_modeling(con_img_mu, con_img_logsigma,con_txt_mu, con_txt_logsigma):
    v=[]
    l=[]
    bs=con_img_mu.shape[0]
    for i in range(bs):
        for j in range(bs):
            v.append(con_img_mu[i:i+1])
            l.append(con_img_logsigma[i:i+1])
    con_img_mu=torch.cat(v)
    con_img_logsigma=torch.cat(l)
    v.clear()
    l.clear()
    for i in range(bs):
        v.append(con_txt_mu)
        l.append(con_txt_logsigma)
    con_txt_mu=torch.cat(v)
    con_txt_logsigma=torch.cat(l)
    z = [con_img_mu] * 1
    # for i in range(5):
    #     eps = torch.randn(con_img_mu.shape[0], con_img_mu.shape[1], con_img_mu.shape[2], device=con_img_mu.device)
    #     z1 = con_img_mu + torch.exp(con_img_logsigma) * eps
    #     z.append(z1)
    image_embeds_z = torch.cat(z)
    zt = [con_txt_mu] * 1
    # for i in range(5):
    #     eps = torch.randn(con_txt_mu.shape[0], con_txt_mu.shape[1], con_txt_mu.shape[2], device=con_txt_mu.device)
    #     z2 = con_txt_mu + torch.exp(con_txt_logsigma) * eps
    #     zt.append(z2)
    text_embeds_z = torch.cat(zt)
    return image_embeds_z,text_embeds_z

def pearson_correlation(sentence_output,video_output):
    text_feat = sentence_output
    cdcr_alpha1 = 0.01
    cdcr_alpha2 = 0.0
    z_a_norm = (text_feat - text_feat.mean(0)) / text_feat.std(0)  # BxD
    # cross-correlation matrix
    B, D = z_a_norm.shape


    # cdcl_video_level 
    video_feat = video_output
    z_b_norm = (video_feat - video_feat.mean(0)) / video_feat.std(0)  # BxD
    c = torch.matmul(z_a_norm.t(),z_b_norm) / B
    #c = torch.einsum('bm,bn->mn', z_a_norm, z_b_norm) / B  # DxD
    # loss
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
    cdcr_loss_video_level = (on_diag * cdcr_alpha1 + off_diag * cdcr_alpha2)
    return cdcr_loss_video_level

def sim_cos(prior_features,posterior_features, prior_mask, posterior_mask):
    # 扩展prior_mask和posterior_mask以便于广播
    # prior_mask_expanded = prior_mask.unsqueeze(2)  
    # posterior_mask_expanded = posterior_mask.unsqueeze(2)
    # prior_features=prior_features*prior_mask_expanded
    # posterior_features=posterior_features*posterior_mask_expanded 
    #将序列特征平均处理成一个向量
    # prior_features=prior_features.mean(dim=1)
    # posterior_features=posterior_features.mean(dim=1)
    # 计算各自的余弦相似度
    norm_prior = F.normalize(prior_features, p=2, dim=-1).squeeze(1)  # [4, 1024]
    norm_posterior = F.normalize(posterior_features, p=2, dim=-1).squeeze(1)  # [4,  1024]
    # norm_prior=prior_features
    # norm_posterior=posterior_features
    prior_cosine_similarity = torch.mm(norm_prior, norm_prior.t())  # [4, 4]
    posterrior_cosine_similarity = torch.mm(norm_posterior, norm_posterior.t())
    sim_mask = torch.triu(
                prior_features.new_ones((prior_features.size(0), prior_features.size(0)), dtype=bool), 1
            )
    prior_cosine_similarity=prior_cosine_similarity.masked_select(sim_mask)
    posterrior_cosine_similarity=posterrior_cosine_similarity.masked_select(sim_mask)
    sim_loss = F.mse_loss(prior_cosine_similarity,posterrior_cosine_similarity)

    return sim_loss*100

def clip_loss(prior_features,posterior_features,logit_scale):
    bt=prior_features.shape[0]
    assert bt==posterior_features.shape[0]
    prior_features=prior_features.reshape(bt,-1)
    posterior_features=posterior_features.reshape(bt,-1)
    # normalized features
    prior_features = prior_features / prior_features.norm(dim=-1, keepdim=True)
    posterior_features = posterior_features / posterior_features.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits_per_struc = logit_scale * prior_features @ posterior_features.T
    logits_per_seq = logits_per_struc.T

    labels = torch.arange(bt, device=prior_features.device, dtype=torch.long)

    loss_struc = F.cross_entropy(logits_per_struc, labels)
    loss_seq = F.cross_entropy(logits_per_seq, labels)
    contra_loss = (loss_struc + loss_seq) / 2

    # return logits_per_struc, logits_per_seq
    return contra_loss*0.1

def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initialize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate through the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2

def get_viterbi_gt(sim_matrix):
        # from L -> R ==>> T -> D
    sim_matrix = sim_matrix.t()
    transition_probability = np.triu(np.full((len(sim_matrix), len(sim_matrix)), 1 / len(sim_matrix)))
    emission_probability = sim_matrix.detach().cpu().numpy()
    obervation_list = np.array(range(sim_matrix.shape[1]))
    import torch.distributed as dist
    # if dist.get_rank() ==0:
    #     print(sim_matrix.shape)
    #     print(obervation_list.shape)
    #     print(transition_probability.shape)
    #     print(emission_probability.shape)
    x, _, _ = viterbi(obervation_list, transition_probability, emission_probability)
    return x

def masked_cosine_similarity(prior_features, posterior_features, prior_mask, posterior_mask):
    # 扩展prior_mask和posterior_mask以便于广播
    prior_mask_expanded = prior_mask.unsqueeze(2)  # [4, 25, 1]
    posterior_mask_expanded = posterior_mask.unsqueeze(1)  # [4, 1, 44]

    # 计算掩码的联合
    joint_mask = prior_mask_expanded & posterior_mask_expanded  # [4, 25, 44]

    # 计算余弦相似度
    norm_prior = F.normalize(prior_features, p=2, dim=-1)  # [4,25, 1024]
    norm_posterior = F.normalize(posterior_features, p=2, dim=-1)  # [4, 44, 1024]
    cosine_similarity = torch.bmm(norm_prior, norm_posterior.transpose(1, 2))  # [4, 25, 44]

    # 使用掩码矩阵来过滤相似度矩阵
    cosine_similarity = cosine_similarity * joint_mask.float().cuda()
    # # 对掩码外的值进行填充
    # cosine_similarity= cosine_similarity.masked_fill(~(joint_mask.bool()), float('-inf'))
    return cosine_similarity

# 定义函数，用于对相似度矩阵按行或按列进行阈值筛选
def apply_threshold_per_row_or_col(similarity_matrix, threshold=0.8, dim=2):
    sorted_similarities, _ = torch.sort(similarity_matrix, dim=dim, descending=True)
    k = int(similarity_matrix.size(dim) * threshold)
    
    if dim == 2:  # 按行处理
        threshold_values = sorted_similarities[:, :, k-1:k]
        mask = similarity_matrix >= threshold_values
    else:  # 按列处理
        threshold_values = sorted_similarities[:, k-1:k, :]
        mask = similarity_matrix >= threshold_values

    return similarity_matrix * mask.float()

def get_extend_mask(text_embeds, visual_src_input, tgt_input_mask, visual_src_attention_mask):

    similarity_matrix = masked_cosine_similarity(text_embeds, visual_src_input, tgt_input_mask, visual_src_attention_mask)
    # 对非零相似度值进行softmax化
    # masked_similarity_matrix = torch.where(similarity_matrix != 0, similarity_matrix, torch.tensor(float('-inf'), device=masked_similarity_matrix.device))
    softmax_similarity_matrix_hang = F.softmax(similarity_matrix, dim=2)
    # 扩展prior_mask和posterior_mask以便于广播
    # prior_mask_expanded = tgt_input['attention_mask'].unsqueeze(2)  # [4, 25, 1]
    prior_mask_expanded = torch.ones(tgt_input_mask.size(0),tgt_input_mask.size(1)).bool().unsqueeze(2)  # [4, 25, 1]
    posterior_mask_expanded = visual_src_attention_mask.unsqueeze(1)  # [4, 1, 44]
    # 计算掩码的联合
    joint_mask1 = prior_mask_expanded & posterior_mask_expanded  # [4, 25, 44]
    # softmax_similarity_matrix_hang=softmax_similarity_matrix_hang.masked_fill(~(joint_mask1.bool()), 0.0)
    # 对按行归一化的相似度矩阵进行阈值筛选
    filtered_softmax_similarity_matrix_hang = apply_threshold_per_row_or_col(softmax_similarity_matrix_hang, dim=2)
    filtered_softmax_similarity_matrix_hang=filtered_softmax_similarity_matrix_hang*joint_mask1.cuda()
    filtered_softmax_similarity_matrix_hang[filtered_softmax_similarity_matrix_hang != 0] = 1

    expanded_mask = filtered_softmax_similarity_matrix_hang.unsqueeze(1).to(visual_src_input.dtype) #使用None和expand方法，将mask从[bsz, seq_len]扩展到[bsz, 1, tgt_seq_len, src_seq_len]。
    inverted_mask = 1.0 - expanded_mask #计算expanded_mask的取反数inverted_mask
    inverted_mask=inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(visual_src_input.dtype).min) 
    return inverted_mask

def sample_similarity(visual_src_input,text_embeds,visual_src_attention_mask, tgt_input_mask):
    # 将视频特征和文本特征展平成单个序列  
    video_features_flat = visual_src_input.contiguous().view(-1, 1024)  # [batch_size * num_video_frames, feature_dim]  
    text_features_flat = text_embeds.contiguous().view(-1, 1024)    # [batch_size * num_text_tokens, feature_dim]
    # 扩展prior_mask和posterior_mask以便于广播
    prior_mask_expanded = visual_src_attention_mask.contiguous().view(-1).unsqueeze(1)  # [4*25, 1]
    posterior_mask_expanded = tgt_input_mask.contiguous().view(-1).unsqueeze(0)  # [1, 4*44]
    # 计算掩码的联合
    joint_mask = prior_mask_expanded & posterior_mask_expanded
    # 计算余弦相似度矩阵
    batch_size=text_embeds.size(0)
    video_frames= visual_src_input.size(1)
    text_tokens=text_embeds.size(1)
    # similarity_matrix = F.cosine_similarity(video_features_flat.unsqueeze(1), text_features_flat.unsqueeze(0), dim=2)
    # 计算余弦相似度
    norm_prior = F.normalize(video_features_flat, p=2, dim=-1)  # [4,25, 1024]
    norm_posterior = F.normalize(text_features_flat, p=2, dim=-1)  # [4, 44, 1024]
    similarity_matrix = torch.mm(norm_prior, norm_posterior.transpose(0, 1))  # [4, 25, 44]
    simplified_sample_similarity_matrix = torch.zeros((batch_size, batch_size))

    for i in range(batch_size):  
        for j in range(batch_size): 
            local_similarity_matrix=similarity_matrix[i*video_frames:(i+1)*video_frames,j*text_tokens:(j+1)*text_tokens]
            local_joint_mask=joint_mask[i*video_frames:(i+1)*video_frames,j*text_tokens:(j+1)*text_tokens]
            local_similarity_matrix_hangmax=torch.mul(local_similarity_matrix,local_joint_mask.float().cuda()).max(dim=1).values
            non_zero_local_similarity_matrix_hangmax=local_similarity_matrix_hangmax[local_similarity_matrix_hangmax!=0.0].mean()
            simplified_sample_similarity_matrix[i,j]=non_zero_local_similarity_matrix_hangmax
    return simplified_sample_similarity_matrix

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt=None, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        if opt is not None:
            self.opt = opt
            self.margin = opt.margin
            self.max_violation = opt.max_violation
        else:
            self.margin = margin
            self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, sims):
        # compute image-sentence score matrix
        # sims = get_sim(im, s)
        diagonal = sims.diag().view(sims.size(0), 1)
        d1 = diagonal.expand_as(sims)
        d2 = diagonal.t().expand_as(sims)

        # compare every diagonal score to sims in its column
        # caption retrieval
        cost_s = (self.margin + sims - d1).clamp(min=0)
        # compare every diagonal score to sims in its row
        # image retrieval
        cost_im = (self.margin + sims - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(sims.size(0)) > .5
        I = torch.tensor(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
# log-sum-exp pooling
class LSEPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==False] = -torch.inf
        sims = torch.logsumexp(sims/self.temperature,dim=-1)
        return sims
    
class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class CrossEn(nn.Module):
    """cross entroy loss"""
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss
######### gating  ##########
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
class GatingMechanism(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc_img = Linear(args.gating_dim * 2, 1)

    def forward(self, x, grid_img_features):


        grid_img_features = torch.mean(grid_img_features, dim=0, keepdim=True)  ## 1*batch*dim
        t, b, c = x.shape
        grid_img_features = grid_img_features.expand(t, b, c)
        merge = torch.cat([x, grid_img_features], dim=-1)

        gate = torch.sigmoid(self.fc_img(merge))  # T B C
        img_features = torch.mul(gate, x)
        return img_features
    
class SLRCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024) :#config`（用于配置模型的参数）和`embed_dim`（特征嵌入的维度，默认为1024）
        super(SLRCLIP, self).__init__()#调用父类（即`nn.Module`）的构造函数
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)
        
        
        # #直接在预训练里翻译
        # self.text_decoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_decoder()
        # # 使用预训练的 MBartForConditionalGeneration 模型作为文本处理模型，只保留解码器部分
        # self.lm_head = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_output_embeddings()
        # # 使用预训练的 MBartForConditionalGeneration 模型作为文本处理模型，只保留输出嵌入
        # self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).model.shared.num_embeddings)))
        
        #跨模态融合编码器
        # self.model = MBartForConditionalGeneration.from_pretrained(#从预训练模型初始化翻译网络，并允许覆盖配置。
        #     '/home/zfc2b/CV-SLT-main/pretrained_models/mBart_de',attention_dropout=0.1,
        #     dropout=0.3
        # )
        # self.text_encoder=self.model.get_encoder() #初始化编码器self.encoder为模型中的编码器
        # self.itm_head=nn.Linear(1024, 2) 
        
        #跨模态bert编码器
        # self.token_type_embeddings = nn.Embedding(2, 1024)
        # self.token_type_embeddings.apply(objectives.init_weights)
        # bert_config = RobertaConfig(
        #     vocab_size=50265,
        #     hidden_size=1024,
        #     num_hidden_layers=6,
        #     num_attention_heads=16,
        #     intermediate_size=1024 * 4,
        #     max_position_embeddings=50,
        #     hidden_dropout_prob=0.1,
        #     attention_probs_dropout_prob=0.1,
        # )
        # self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(2)])
        # self.cross_modal_image_layers.apply(objectives.init_weights)
        # self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(2)])
        # self.cross_modal_text_layers.apply(objectives.init_weights)

        #PED概率分布编码器的操作
        # self.con_img_gau_encoder = DisTrans(1024, 16)
        # self.con_txt_gau_encoder = DisTrans(1024, 16)
        # self.con_img_gau_encoder.apply(objectives.init_weights)
        # self.con_txt_gau_encoder.apply(objectives.init_weights)
        # self.negative_scale = 1/200
        # self.shift = 4
        # self.temp = nn.Parameter(torch.ones([]) * 0.07)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        # self.cross_modal_image_pooler=Pooler(1024)
        # self.cross_modal_text_pooler=Pooler(1024)
        # self.cross_modal_image_pooler.apply(objectives.init_weights)
        # self.cross_modal_text_pooler.apply(objectives.init_weights)
        #直接提取cls-token后的头部全连接
        # self.lm_head = make_head(1024, 1024, 'linear')
        # # self.lm_head1 = make_head(1024, 1024, 'linear')
        # self.lm_head2=make_head(1024, 1024, 'identy')
        # self.loss_fct=CrossEn()

        #经过重参数化然后文本和视频进行匹配
        # self.itm_score = ITMHead(512*2)
        # self.itm_score.apply(objectives.init_weights)

        #TAB时间对齐模块初始化
        # self.centerK=9
        # # initialize the weight center
        # self.weight_center = nn.Parameter(torch.empty(self.centerK, 1024))
        # nn.init.normal_(self.weight_center, std=64 ** -0.5)
        # # initialize the embedding center
        # self.emb_center = nn.Parameter(torch.empty(self.centerK, 1024))
        # nn.init.normal_(self.emb_center, std=64 ** -0.5)
        # self.loss_fct=CrossEn()
        # #一层
        # self.actionClip = TransformerClip(width=1024, layers=1, heads=16, )
        # self.sim_type ="seqTransf"
        # # positional embedding for temporal transformer
        # self.frame_position_embeddings = nn.Embedding(77, 1024)
        # # 4-layer temporal transformer
        # self.transformerClip = TransformerClip(width=1024,
        #                                            layers=4,
        #                                            heads=16, )
        # self.center_proj = 'TAB_TDB'
        # self.temporal_proj = 'sigmoid_selfA'
        #  # initialize for difference-level attention
        # self.frame2t_attention = TransformerClip(width=1024, layers=1, heads=16, )
        # self.temporal_type = 'TDB'
        # self.type_position_embeddings = nn.Embedding(2, 1024)
        # self.sigmoid = torch.nn.Sigmoid()
        # self.trans_layernorm = torch.nn.LayerNorm(1024)

        #TIB文本视频交互模块
        # self.loss_fct=CrossEn()
        # self.frame_position_embeddings = nn.Embedding(120, 1024)
        # self.transformerClip = TransformerClip(width=1024, layers=1,
        #                                            heads=16, )
        # for coarse-grained constrast weights
        # self.global_mat_weight = nn.parameter.Parameter(torch.eye(1024), requires_grad=True)

        # Define the Contrastive Loss Criterion，signcl论文里的损失函数
        # self.cl_criterion = SignCL(max_distance=32.0, pos_samples=2, neg_samples=4)

        #token聚合操作模块
        # self.downsample=nn.Linear(1024,512)
        # self.relu = nn.ReLU()  
        # # self.upsample=nn.Linear(512,1024)
        # self.t_ctm0 = CTM(sample_ratio=0.5, embed_dim=512, dim_out=512, k=3)
        # self.t_block0 = TCBlock(dim=512, num_heads=8)
        # self.v_ctm0 = CTM(sample_ratio=0.7, embed_dim=512, dim_out=512, k=3)
        # self.v_block0 = TCBlock(dim=512, num_heads=8)
        # # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        # # self.loss_fct = CrossEn()
        # self.interaction ='wti'
        # self.text_weight_fc0 = nn.Sequential(
        #     nn.Linear(512, 2 * 512), nn.ReLU(inplace=True),
        #     nn.Linear(2 * 512, 1))
        # self.video_weight_fc0 = nn.Sequential(
        #     nn.Linear(512, 2 * 512), nn.ReLU(inplace=True),
        #     nn.Linear(2 * 512, 1))

        # self.text_weight_fc0 = nn.Sequential(
        #     nn.Linear(512, 2 * 512), nn.ReLU(inplace=True),
        #     nn.Linear(2 * 512, 1))
        # self.video_weight_fc0 = nn.Sequential(
        #     nn.Linear(512, 2 * 512), nn.ReLU(inplace=True),
        #     nn.Linear(2 * 512, 1))

        self.gloss_tokenizer = self.model_images.gloss_tokenizer

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # 用于初始化分类头的可训练参数，用于缩放logits，初始化为`np.log(1 / 0.07)`，即0.2236
        # self.con_loss=ContrastiveLoss(margin=0.2)
        # self.sim_embt = GraphEmbt(1024, 256)
        self.sim_embvt = GraphEmbv(1024, 256)
        self.sim_embv = GraphEmbv(1024, 256)
        # self.sim_enc = EncoderSimilarity(1024, 256, 3)
    def compute_vtm(self, text_embeds, text_atts, image_embeds, image_atts, sim_i2t, sim_t2i):
        device = text_embeds.device

        # ====== positive pairs =======
        attention_mask = torch.cat([image_atts, text_atts], dim=1)
        embedding_output_pos = torch.cat([image_embeds, text_embeds], dim=1)

        encoder_outputs_pos = self.text_encoder(inputs_embeds=embedding_output_pos.cuda(),
                                                     attention_mask=attention_mask.cuda(),
                                                     return_dict=True
                                                    )

        # ====== negative pairs =======
        bs = text_embeds.shape[0] 

        local_rank = 0
        b_start, b_end = bs * local_rank, bs * (local_rank + 1)

        with torch.no_grad():
            weights_v2t = sim_i2t[:,b_start:b_end]
            weights_t2v = sim_t2i[:,b_start:b_end]
   
            # never select self as negative
            weights_v2t.fill_diagonal_(-np.Inf)
            weights_t2v.fill_diagonal_(-np.Inf)

            weights_v2t = F.softmax(weights_v2t, dim=1)
            weights_t2v = F.softmax(weights_t2v, dim=1)

        # select a negative image for each text
        # FIXME to optimize using indexing operations
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2v[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_v2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text_atts, text_atts_neg],dim=0)     

        video_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        video_atts_all = torch.cat([image_atts,image_atts],dim=0)

        attention_mask_all = torch.cat([video_atts_all, text_atts_all], dim=1)
        embedding_output_all = torch.cat([video_embeds_all, text_embeds_all], dim=1)

        # forward negative pairs via cross encoder
        encoder_outputs_neg = self.text_encoder(inputs_embeds=embedding_output_all.cuda(),
                                                     attention_mask=attention_mask_all.cuda(),
                                                     return_dict=True
                                                    )

        vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                   encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)
        vtm_logits = self.itm_head(vl_embeddings)            

        vtm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)], dim=0).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)     

        return vtm_loss, vtm_logits, vtm_labels 

    def action_level(self, text_feat, video_feat, text_mask, video_mask, t_token_dict=None, v_token_dict=None):
        if self.interaction == 'wti':
            text_weight = self.text_weight_fc0(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
            text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

            video_weight = self.video_weight_fc0(video_feat).squeeze(2)  # B x N_v x D -> B x N_v
            video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])

        text_sum = text_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        if self.interaction == 'wti':  # weighted token-wise interaction
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
            t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
            v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])

            _retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        else:
            # max for video token
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
            t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
            v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))
            _retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        return _retrieve_logits, _retrieve_logits.T, retrieve_logits
    
    def get_extra_TAB_embedding(self, embedding_out, attention_mask):
        """ obtain frame embedding concentrated with temporal embedding
        Args:
            embedding_out: token embedding
            attention_mask: frame embedding
        Returns:
            embedding_out: token embedding with temporal enhancing
            attention_mask: frame embedding with temporal enhancing
        """
        large_position_d = torch.arange(start=0, end=embedding_out.size()[1], step=2, dtype=torch.long,
                                        device=embedding_out.device)
        large_embedding_out = embedding_out[:, large_position_d, :]  # bs * 6 * 512
        large_attention_mask = attention_mask[:, large_position_d]

        # embedding_out: bs * seq * 512 | local_out: bs * seq * (k + 1)
        if self.center_proj == 'TAB' or self.center_proj == 'TAB_TDB':

            # sample in the large frame rate

            # obtain the attention mask of large frame rate
            large_attention_mask_span = large_attention_mask.squeeze(-1)
            large_attention_mask_span = large_attention_mask_span.squeeze(-1)

            # prepare the position embedding and store the input embedding
            seq_length = large_embedding_out.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=large_embedding_out.device)
            position_ids = position_ids.unsqueeze(0).expand(large_embedding_out.size(0), -1)
            TAB_position_embedding = self.frame_position_embeddings(position_ids)
            large_embedding_out_original = large_embedding_out

            if self.center_proj == 'TAB_TDB':
                # shared TDB is adopted to insert difference-enhanced token
                large_embedding_out, TAB_position_embedding, TAB_type_embedding, large_attention_mask_span = self.temporal_difference_block(
                    large_embedding_out, large_attention_mask_span)
                large_embedding_out = large_embedding_out + TAB_position_embedding + TAB_type_embedding
            else:
                large_embedding_out = large_embedding_out + TAB_position_embedding  # batch_size * 12 * 512

            extended_video_mask = (1.0 - large_attention_mask_span.unsqueeze(1)) * -1000000.0  # batch_size * 1* 12
            extended_video_mask = extended_video_mask.expand(-1, large_attention_mask_span.size(1),
                                                             -1)  # batch_size * 12 * 12

            # adopt 1-layer temporal transformer to encode representation
            large_embedding_out = large_embedding_out.permute(1, 0, 2)  # NLD -> LND # 12 * batch_size * 512
            large_embedding_out = self.actionClip(large_embedding_out, extended_video_mask)  # 12 * batch_size * 512
            large_embedding_out = large_embedding_out.permute(1, 0, 2)  # LND -> NLD # batch_size * 12 * 512

            # adopt the output of frame token if use TAB_TDB
            if self.center_proj == 'TAB_TDB':
                frame_position_id = torch.arange(start=0, end=large_embedding_out.size()[1], step=2, dtype=torch.long,
                                                 device=large_embedding_out.device)
                large_embedding_out = large_embedding_out[:, frame_position_id, :]

            # concat the original embedding and encoded embedding with temporal correlations
            large_embedding_out = large_embedding_out + large_embedding_out_original
            embedding_out = torch.cat((embedding_out, large_embedding_out), 1)
            attention_mask = torch.cat((attention_mask, large_attention_mask), 1)

        return embedding_out, attention_mask
    
    def temporal_difference_block(self, visual_output, video_mask):
        """Calculate difference-enhanced token and inset into frame token
        Args:
            visual_output: embedding
            video_mask: video mask
        Returns:
            visual_output: frame representation
            frame_position_embeddings: position embedding
            type_embedding: type embedding
            temporal_video_mask: attention mask
        """

        seq_length = visual_output.size(1) # 12

        # obtain the positional embedding
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device) # 12
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1) # batch_size * 12
        frame_position_embeddings = self.frame_position_embeddings(position_ids) # batch_size * 12 * 512

        # obtain the type embedding to indicate the frame token and difference-enhanced token
        video_ids = torch.ones_like(position_ids)
        videoDif_ids = torch.zeros_like(position_ids)
        video_type_embedding = self.type_position_embeddings(video_ids)
        videoDif_type_embedding = self.type_position_embeddings(videoDif_ids)

        # adopt temporal_proj == sigmoid_mlp for mlp transformation
        # adopt temporal_proj == sigmoid_selfA for difference-level attention
        # adopt temporal_proj == default to use subtraction directly

        # batch size * 11 * 512
        dif_visual_output = visual_output[:, 1: seq_length, :] - visual_output[:, 0: seq_length - 1, :]
        if self.temporal_proj == 'sigmoid_mlp':
            # adopt sigmoid to transform into [-1, 1]
            dif_visual_output = 2 * self.sigmoid(self.trans_layernorm(dif_visual_output @ self.frame2t_projection)) - 1

        elif self.temporal_proj == 'sigmoid_selfA':
            # batch_size * 11 * 512
            dif_visual_output = dif_visual_output + frame_position_embeddings[:, 1:seq_length, :]
            trans_video_mask = video_mask[:,1:seq_length]
            # batch_size * 1* 11
            extend_trans_video_mask = (1.0 - trans_video_mask.unsqueeze(1)) * -1000000.0
            # batch_size * 11 * 11
            extend_trans_video_mask = extend_trans_video_mask.expand(-1, trans_video_mask.size(1), -1)

            dif_visual_output = dif_visual_output.permute(1, 0, 2)  # NLD -> LND # 11 * batch_size * 512
            dif_visual_output = self.frame2t_attention(dif_visual_output, extend_trans_video_mask)
            dif_visual_output = dif_visual_output.permute(1, 0, 2)  # LND -> NLD # batch_size * 11 * 512

            dif_visual_output = 2 * self.sigmoid(self.trans_layernorm(dif_visual_output)) - 1

        # batch size * (12+11) * 512
        visual_middle = torch.cat((visual_output, dif_visual_output), 1)
        # batch size * (12+12) * 512
        frame_position_embeddings_middle = torch.cat((frame_position_embeddings, frame_position_embeddings), 1)
        temporal_video_mask_middle = torch.cat((video_mask, video_mask), 1).cuda()
        type_embedding_middle = torch.cat((video_type_embedding, videoDif_type_embedding), 1)

        # obtain the correct index to insert difference-enhanced token
        seq1_indices = torch.arange(start=0, end=seq_length, step=1, dtype=torch.long)
        seq2_indices = torch.arange(start=seq_length, end=2 * seq_length - 1, step=1, dtype=torch.long)
        seq_indices = torch.stack((seq1_indices[0], seq2_indices[0]))
        for i in range(1, seq_length - 1):
            seq_indices = torch.cat((seq_indices, seq1_indices[i].view(1), seq2_indices[i].view(1)))
        seq_indices = torch.cat((seq_indices, seq1_indices[seq_length - 1].view(1))).cuda()

        # insert difference-enhanced token between every adjacent frame token
        visual_output = visual_middle.index_select(1, seq_indices)
        frame_position_embeddings = frame_position_embeddings_middle.index_select(1, seq_indices)
        temporal_video_mask = temporal_video_mask_middle.index_select(1, seq_indices)
        type_embedding = type_embedding_middle.index_select(1, seq_indices)

        return visual_output, frame_position_embeddings, type_embedding, temporal_video_mask
    
    def get_TAB_embedding(self, embedding_out, attention_mask, type='default'):
        """ obtain aligned embedding for video and text
        Args:
            embedding_out: token embedding
            attention_mask: frame embedding
        Returns:
            cluster_embedding: aligned embedding
        """
        if type == 'visual':
            embedding_out, attention_mask = self.get_extra_TAB_embedding(embedding_out, attention_mask)


        soft_weight = F.softmax(embedding_out @ self.weight_center[0:self.centerK].t(), 2)


        cluster_embedding = soft_weight.unsqueeze(3) * (embedding_out.unsqueeze(2) - self.emb_center[0:self.centerK])
        cluster_embedding = torch.sum(cluster_embedding * attention_mask.cuda(), 1)

        cluster_embedding = cluster_embedding / cluster_embedding.norm(dim=-1, keepdim=True)
        cluster_embedding = torch.mean(cluster_embedding, dim=1)
        cluster_embedding = cluster_embedding / cluster_embedding.norm(dim=-1, keepdim=True)

        return cluster_embedding
    def calc_TAB_loss(self, sequence_hidden_output, visual_output, attention_mask, video_mask):
        """ calculate TAB loss
         Args:
             sequence_hidden_output: token embedding
             visual_output: frame embedding
             attention_mask: caption mask
             video_mask: video mask
         Returns:
             sim_loss: loss for optimization
         """

        # obtain the aligned video representation
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_mask_un = video_mask_un.unsqueeze(-1)
        cluster_visual_output = self.get_TAB_embedding(visual_output, video_mask_un, type='visual')

        # obtain the aligned text representation
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        # attention_mask_un[:, 0, :] = 0.
        attention_mask_un = attention_mask_un.unsqueeze(-1)
        cluster_sequence_output = self.get_TAB_embedding(sequence_hidden_output, attention_mask_un, type='sequence')

        # calculate the similarity
        logit_scale = self.logit_scale.exp()
        sim_matrix = logit_scale * torch.matmul(cluster_sequence_output, cluster_visual_output.t())

        # text-to-video loss
        sim_loss1 = self.loss_fct(sim_matrix)

        # video-to-text loss
        sim_loss2 = self.loss_fct(sim_matrix.T)

        sim_loss = (sim_loss1 + sim_loss2) / 2
        return sim_loss

    def get_model_txt(self):#定义一个方法来获取文本处理模型
        return self.model_txt
    
    def generate_gumbel_sort_gt(self,preds_label=None):
        gt_label = torch.max(preds_label, dim=-1)[1]  # [b,t,n](one hot) -> [b,t](index)
        gt_label = torch.sort(gt_label, dim=-1)[0]  # [b,t](index) ->sort-> [b,t](ground truth)
        return gt_label
    
    def generate_gumbel_viterbi_gt(self,sim_matrices = None):
        # sim_matrix [b,t,n]
        b,t,n = sim_matrices.shape
        gt = []
        for sim_matrix in sim_matrices:
            x = get_viterbi_gt(sim_matrix)
            gt.append(torch.from_numpy(x).unsqueeze(dim=0).long().to(sim_matrices.device))
        gt = torch.cat(gt, dim=0)
        return gt

    def flip_similarity_softmax(self, sequence_output, visual_output, attention_mask, video_mask,v_list,t_list, sequence_hidden_aug=None,text_mask_aug=None):

        video_mask = (video_mask == 1).cuda()
        attention_mask=(attention_mask==1).cuda()
        # text_mask_aug=(text_mask_aug==1)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        # visual_output = visual_output.squeeze(1)

        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        # sequence_output = sequence_output.squeeze(1)

        batch_size,v_len=visual_output.shape[0],visual_output.shape[1]
        batch_size_t,t_len=sequence_output.shape[0],sequence_output.shape[1]

        # sequence_hidden_aug = sequence_hidden_aug / sequence_hidden_aug.norm(dim=-1, keepdim=True)
        # sequence_hidden_aug = sequence_hidden_aug.squeeze(1)


        logit_scale = self.logit_scale.exp()
        i2t_sim=torch.einsum("ais,bjs->abij", [visual_output, sequence_output])
        # i2t_sim_aug=torch.einsum("ais,bjs->abij", [visual_output, sequence_hidden_aug])

        # after_softmax_i2t = torch.nansum(i2t_sim * torch.softmax(i2t_sim/0.07, dim=3), dim=3)
        # # after_softmax_i2t , _ = torch.max(i2t_sim, dim=3)
        # video_mask_extend=video_mask.unsqueeze(1).repeat(1,batch_size_t,1)
        # after_softmax_i2t[~video_mask_extend]=0
        # # I2T_sim = logit_scale*torch.nansum(after_softmax_i2t, dim=-1)/torch.sum(video_mask_extend,dim=-1)
        # after_softmax_i2t[after_softmax_i2t==0] = -torch.inf
        # I2T_sim = torch.logsumexp(after_softmax_i2t/0.1,dim=-1)
        #置信度分数对比模块
        v_list=v_list.permute(1, 0, 2,3)
        # 计算余弦相似度
        norm_prior = F.normalize(v_list, p=2, dim=-1)  # [4,4,25, 1024]
        norm_posterior = F.normalize(t_list, p=2, dim=-1)  # [4,4, 44, 1024]
        similarity_matrix = torch.matmul(norm_prior, norm_posterior.transpose(3, 2))  # [4,4, 25, 44]
        after_softmax_i2t = torch.nansum(i2t_sim * torch.softmax(similarity_matrix/0.07, dim=3), dim=3)*0.5+torch.nansum(i2t_sim * torch.softmax(i2t_sim/0.07, dim=3), dim=3)*0.5
        # after_softmax_i2t , _ = torch.max(i2t_sim, dim=3)
        video_mask_extend=video_mask.unsqueeze(1).repeat(1,batch_size_t,1)
        after_softmax_i2t[~video_mask_extend]=0
        # I2T_sim = logit_scale*torch.nansum(after_softmax_i2t, dim=-1)/torch.sum(video_mask_extend,dim=-1)
        after_softmax_i2t[after_softmax_i2t==0] = -torch.inf
        I2T_sim = torch.logsumexp(after_softmax_i2t/0.1,dim=-1)


        # v_list[~video_mask_extend]=0
        # v_list[v_list==0] = -torch.inf
        # softmax_v_list=torch.softmax(v_list/0.07, dim=2)
        # I2T_sim = torch.nansum(after_softmax_i2t*softmax_v_list, dim=-1)

        # after_softmax_t2i = torch.nansum(i2t_sim_aug * torch.softmax(i2t_sim_aug/0.07, dim=2), dim=2)
        # text_mask_extend2=text_mask_aug.unsqueeze(0).repeat(batch_size,1,1)
        # after_softmax_t2i[~text_mask_extend2]=0

        after_softmax_t2i = torch.nansum(i2t_sim * torch.softmax(similarity_matrix/0.07, dim=2), dim=2)*0.5+ torch.nansum(i2t_sim * torch.softmax(i2t_sim /0.07, dim=2), dim=2)*0.5
        # after_softmax_t2i, _ = torch.max(i2t_sim, dim=2)
        text_mask_extend2=attention_mask.unsqueeze(0).repeat(batch_size,1,1)
        after_softmax_t2i[~text_mask_extend2]=0

        # T2I_sim = logit_scale*torch.nansum(after_softmax_t2i*text_mask_extend2, dim=-1)/torch.sum(text_mask_extend2,dim=-1)
        after_softmax_t2i[after_softmax_t2i==0] = -torch.inf
        T2I_sim = torch.logsumexp(after_softmax_t2i/0.1,dim=-1)
        #置信度分数对比模块
        # t_list[~text_mask_extend2]=0
        # t_list[t_list==0] = -torch.inf
        # softmax_t_list=torch.softmax(t_list/0.07, dim=2)
        # T2I_sim =torch.nansum(after_softmax_t2i*softmax_t_list, dim=-1)

        return I2T_sim,T2I_sim
    
    @property  #使用Python的装饰器，将下一个方法转换为属性
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states #定义一个属性方法来获取文本模型的隐藏状态。
    
    def forward(self, src_input, tgt_input): #定义前向传播函数，接收两个输入：`src_input`（源输入）和`tgt_input`（目标输入，通常是文本）
        image_features,visual_src_input,visual_src_attention_mask,frames = self.model_images(src_input)
        text_features, self.encoder_hidden_states,text_embeds = self.model_txt(tgt_input)


        #直接在预训练里翻译
        # decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        # #使用 shift_tokens_right 函数将 tgt_input['input_ids'] 向右移动一个位置，将输入文本的第一个元素替换为特殊符号，通常是一个特殊的标记，表示序列的开始，作为文本解码器的输入
        # decoder_out = self.text_decoder(
        #             input_ids = decoder_input_ids, # decoder_input_ids 的形状应为 (B, 64)，表示文本解码器的输入
        #             attention_mask = tgt_input['attention_mask'].cuda(), # tgt_input['attention_mask'] 的形状应为 (B, 64)，指示解码器在生成输出时应该关注输入序列中的哪些部分
        #             encoder_hidden_states = visual_src_input[:,1:,:], # encoder_hidden_states 的形状应为 (B, 65, 1024)，解码器会使用这些编码器隐藏状态来生成输出
        #             encoder_attention_mask = visual_src_attention_mask[:,1:].cuda(),# encoder_attention_mask 的形状应为 (B, 65)，指示解码器在生成输出时应该关注编码器输出中的哪些部分
        #             return_dict = True,
        #             )

        # lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias
        #概率分布编码器
        # extend_image_masks= _expand_mask(visual_src_attention_mask, visual_src_input.dtype)
        # extend_text_masks=_expand_mask(tgt_input['attention_mask'], visual_src_input.dtype)
        # con_img_mu, con_img_logsigma,_= self.con_img_gau_encoder(visual_src_input.cuda(), mask=extend_image_masks.cuda())
        # con_txt_mu, con_txt_logsigma, _ = self.con_txt_gau_encoder(text_embeds.cuda(), mask=extend_text_masks.cuda())

        #实现计算每个样本间token的相似度
        # video_text_similarity_matrix=sample_similarity(visual_src_input[:,1:,:],text_embeds,visual_src_attention_mask[:,1:], tgt_input['attention_mask']).cuda()
        # text_video_similarity_matrix=sample_similarity(text_embeds,visual_src_input[:,1:,:], tgt_input['attention_mask'],visual_src_attention_mask[:,1:]).cuda()
        # M_loss_t2v = self.loss_fct(text_video_similarity_matrix * self.logit_scale.exp())
        # M_loss_v2t = self.loss_fct(video_text_similarity_matrix * self.logit_scale.exp())
        # M_loss = (M_loss_t2v + M_loss_v2t) / 2

        # #计算视觉和文本token的相似度,根据相似度进行筛选注意力mask的关注帧
        # extend_image_masks1=get_extend_mask(text_embeds, visual_src_input[:,1:,:], tgt_input['attention_mask'], visual_src_attention_mask[:,1:])
        # extend_text_masks1=get_extend_mask( visual_src_input[:,1:,:], text_embeds, visual_src_attention_mask[:,1:],tgt_input['attention_mask'])
        # #跨模态编码器
        # extend_image_masks= _expand_mask(visual_src_attention_mask[:,1:], visual_src_input.dtype)
        # extend_text_masks=_expand_mask(tgt_input['attention_mask'], visual_src_input.dtype)
        # # extend_image_masks1= _expand_mask(visual_src_attention_mask[:,1:], visual_src_input.dtype,tgt_len=tgt_input['attention_mask'].size()[-1])
        # # extend_text_masks1=_expand_mask(tgt_input['attention_mask'], visual_src_input.dtype,tgt_len=visual_src_attention_mask[:,1:].size()[-1])
        # text_embeds, image_embeds = (
        #     text_embeds.cuda() + self.token_type_embeddings(torch.zeros_like(tgt_input['attention_mask'].cuda())).cuda(),
        #     visual_src_input[:,1:,:].cuda()
        #     + self.token_type_embeddings(
        #         torch.full_like(visual_src_attention_mask[:,1:].cuda(), 1)
        #     ),
        # )
        # x, y = text_embeds, image_embeds
        # for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
        #     x1 = text_layer(x, y, extend_text_masks, extend_image_masks1)
        #     y1 = image_layer(y, x, extend_image_masks, extend_text_masks1)
        #     x, y = x1[0], y1[0]

        #直接在预训练里翻译
        # decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        # #使用 shift_tokens_right 函数将 tgt_input['input_ids'] 向右移动一个位置，将输入文本的第一个元素替换为特殊符号，通常是一个特殊的标记，表示序列的开始，作为文本解码器的输入
        # decoder_out = self.text_decoder(
        #             input_ids = decoder_input_ids, # decoder_input_ids 的形状应为 (B, 64)，表示文本解码器的输入
        #             attention_mask = tgt_input['attention_mask'].cuda(), # tgt_input['attention_mask'] 的形状应为 (B, 64)，指示解码器在生成输出时应该关注输入序列中的哪些部分
        #             encoder_hidden_states = x, # encoder_hidden_states 的形状应为 (B, 65, 1024)，解码器会使用这些编码器隐藏状态来生成输出
        #             encoder_attention_mask = tgt_input['attention_mask'].cuda(),# encoder_attention_mask 的形状应为 (B, 65)，指示解码器在生成输出时应该关注编码器输出中的哪些部分
        #             return_dict = True,
        #             )

        # lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        # con_img_mu, con_img_logsigma,_= self.con_img_gau_encoder(y.cuda(), mask=extend_image_masks.cuda())
        # con_txt_mu, con_txt_logsigma, _ = self.con_txt_gau_encoder(x.cuda(), mask=extend_text_masks.cuda())
        #经过均值方差乘以高斯噪声再进行clip_loss
        # eps1 = torch.randn(con_img_mu.shape[0], con_img_mu.shape[1], con_img_mu.shape[2], device=con_img_mu.device)
        # z1 = con_img_mu + torch.exp(con_img_logsigma) * eps1*0.01
        # z1=self.lm_head(z1[:,0,:])
        # eps2 = torch.randn(con_txt_mu.shape[0], con_txt_mu.shape[1], con_txt_mu.shape[2], device=con_txt_mu.device)
        # z2 = con_txt_mu + torch.exp(con_txt_logsigma) * eps2*0.01
        # con_txt_mu1=z2[torch.arange(z2.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        # z2=self.lm_head2(con_txt_mu1)
        # z1 = z1 / z1.norm(dim=-1, keepdim=True)
        # z2 = z2 / z2.norm(dim=-1, keepdim=True)
        # # calculate the similarity
        # logit_scale1 = self.logit_scale.exp()         
        # sim_matrix = logit_scale1 * torch.matmul(z1, z2.t())
        # # sim_matrix = sim_matrix * F.softmax(sim_matrix/1000, dim=0)*len(sim_matrix)#相应的DSL来修正预测的相似度评分
        # # text-to-video loss
        # sim_loss1 = self.loss_fct(sim_matrix)
        # # video-to-text loss
        # sim_loss2 = self.loss_fct(sim_matrix.T)
        # sim_loss = (sim_loss1 + sim_loss2) / 2

        #视觉和文本均值方差的对比损失，类似info_nce
        # # con_img_mu1= self.avgpool(con_img_mu.transpose(1, 2)).view(con_img_mu.size(0), 1, -1)#平均池化
        # # con_img_mu1=self.cross_modal_image_pooler(con_img_mu)
        # con_img_mu1=self.lm_head(con_img_mu[:,0,:])
        # # con_img_mu1=con_img_mu1/ con_img_mu1.norm(dim=-1, keepdim=True)
        # con_img_logsigma1=self.lm_head1(con_img_logsigma[:,0,:])
        # # con_img_logsigma1=con_img_logsigma1/ con_img_logsigma1.norm(dim=-1, keepdim=True)
        # # con_img_logsigma1=self.avgpool(con_img_logsigma.transpose(1, 2)).view(con_img_logsigma.size(0), 1, -1)#平均池化
        # # con_img_logsigma1=self.cross_modal_image_pooler(con_img_logsigma)
        # # con_txt_mu1=self.avgpool1(con_txt_mu.transpose(1, 2)).view(con_txt_mu.size(0), 1, -1)#平均池化
        # # con_txt_mu1=self.cross_modal_text_pooler(con_txt_mu1)
        # # con_txt_logsigma1=self.avgpool1(con_txt_logsigma.transpose(1, 2)).view(con_txt_logsigma.size(0), 1, -1)#平均池化
        # # con_txt_logsigma1=self.cross_modal_text_pooler(con_txt_logsigma1)
        # con_txt_mu1=con_txt_mu[torch.arange(con_txt_mu.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        # con_txt_mu1=self.lm_head2(con_txt_mu1)
        # # con_txt_mu1=con_txt_mu1/ con_txt_mu1.norm(dim=-1, keepdim=True)
        # con_txt_logsigma1=con_txt_logsigma[torch.arange(con_txt_logsigma.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        # con_txt_logsigma1=self.lm_head2(con_txt_logsigma1)
        # # con_txt_logsigma1=con_txt_logsigma1/ con_txt_logsigma1.norm(dim=-1, keepdim=True)
        # #计算W2距离
        # bs=visual_src_input.shape[0]
        # W2_distance, mu_distance = Wasserstein2(con_img_mu1,  torch.exp(con_img_logsigma1),con_txt_mu1,  torch.exp(con_txt_logsigma1))
        # similarity = (-self.negative_scale * W2_distance + self.shift) / self.temp
        # similarity=similarity.squeeze(-1)
        # labels = torch.arange(bs).to(similarity.device)
        # clip_loss = ((F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)) / 2)*1.0


        #视频文本的匹配损失
        # image_embeds_z,text_embeds_z=gaussian_modeling(con_img_mu, con_img_logsigma,con_txt_mu, con_txt_logsigma)
        # # avg_image_feats = self.avgpool(image_embeds_z.transpose(1, 2)).view(image_embeds_z.size(0), 1, -1)#平均池化
        # # cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        # cls_feats_image=self.lm_head(image_embeds_z[:,0,:])
        # # avg_text_feats = self.avgpool1(text_embeds_z.transpose(1, 2)).view(text_embeds_z.size(0), 1, -1)#平均池化
        # # cls_feats_text = self.cross_modal_text_pooler(avg_text_feats)
        # b=tgt_input['input_ids'].shape[0]
        # # 初始化一个空的tensor来存储结果  
        # repeated_indices = torch.empty(0, dtype=torch.long)  # 使用正确的dtype  
        # # 循环复制并拼接  
        # for _ in range(b):  
        #     repeated_indices = torch.cat((repeated_indices, tgt_input['input_ids'].argmax(dim=-1)), dim=0) 
        # cls_feats_text=text_embeds_z[torch.arange(text_embeds_z.shape[0]), repeated_indices]
        # cls_feats_text=self.lm_head2(cls_feats_text)
        # # cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        # cls_feats_text=cls_feats_text.unsqueeze(1)
        # cls_feats_image=cls_feats_image.unsqueeze(1)
        # cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=1)
        # cls_feats = self.avgpool(cls_feats.transpose(1, 2)).view(cls_feats.size(0), 1, -1)#平均池化
        # cls_feats= self.cross_modal_image_pooler(cls_feats)

        # bs=con_img_mu.shape[0]
        # itm_labels= torch.eye(bs).flatten().to(torch.float32) .to(cls_feats)
        # itm_logits = self.itm_score(cls_feats)
        # itm_logits = itm_logits.reshape((1), -1, itm_logits.shape[-1])
        # itm_loss = []
        # for i in range(1):
        #     itm_loss.append(F.cross_entropy(itm_logits[i], itm_labels.long()))   
        # itm_loss = sum(itm_loss) / (1)
        # # itm_logits = torch.mean(itm_logits, dim=0)

        #TAB时间对齐模块
        # calculate TAB loss
        # TAB_loss = self.calc_TAB_loss(self.encoder_hidden_states, frames, tgt_input['attention_mask'], visual_src_attention_mask[:,1:])

        #TIB文本视频交互模块
        #视觉特征算损失之前再一次temporal encoder
        # Sequential type: Transformer Encoder
        # visual_output=visual_src_input[:,1:,:]
        # visual_output_original = visual_output
        # seq_length = visual_output.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        # position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        # # position_ids : [bs, num_frames, dim]
        # frame_position_embeddings = self.frame_position_embeddings(position_ids)
        # visual_output = visual_output + frame_position_embeddings
        # visual_src_attention_mask1=visual_src_attention_mask[:,1:]
        # extended_video_mask = (1.0 - visual_src_attention_mask1.unsqueeze(1)) * -1000000.0
        # extended_video_mask = extended_video_mask.expand(-1, visual_src_attention_mask1.size(1), -1)
        # visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        # visual_output = self.transformerClip(visual_output, extended_video_mask)
        # visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        # visual_output = visual_output + visual_output_original

        # #strong sentence-frame score 
        # visual_output_norm = visual_output / visual_output.norm(dim=-1, keepdim=True)
        # frame_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)
        # sentence2frames_sim = torch.matmul(text_features, frame_features.permute(0, 2, 1)) #BT,D BDF -->B,BT,F  # / 1e-2
        # sentence2frames_sim = torch.diagonal(sentence2frames_sim).T #FB-->BF
        # sentence2frames_sim = sentence2frames_sim.unsqueeze(2)
        # sentence2frames_sim = sentence2frames_sim.detach()

        # visual_output_adapt = (visual_output_norm * sentence2frames_sim.div(0.1).softmax(1)).sum(1)
        # visual_output_adapt = visual_output_adapt / visual_output_adapt.norm(dim=-1, keepdim=True)
        # sentence_frame_strong_logits = self.logit_scale.exp() * torch.matmul(text_features, visual_output_adapt.t())
        
        #自己加一个另外的strong video-words score
        # text_embeds1= text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # video2words_sim = torch.matmul(image_features, text_embeds1.permute(0, 2, 1)) #BT,D BDF -->B,BT,F  # / 1e-2
        # video2words_sim  = torch.diagonal(video2words_sim).T #FB-->BF
        # video2words_sim  = video2words_sim.unsqueeze(2)
        # video2words_sim  = video2words_sim.detach()

        # text_output_adapt = (text_embeds * video2words_sim.div(0.1).softmax(1)).sum(1)
        # text_output_adapt = text_output_adapt / text_output_adapt.norm(dim=-1, keepdim=True)
        # video2words_strong_logits = self.logit_scale.exp() * torch.matmul(image_features, text_output_adapt.t())

        # sim_strong_loss1 = self.loss_fct(sentence_frame_strong_logits)
        # sim_strong_loss2 = self.loss_fct(sentence_frame_strong_logits.T)
        # sim_strong_loss = (sim_strong_loss1 + sim_strong_loss2) / 2

        #token聚合操作的模块
        # text_feat,text_mask =self.encoder_hidden_states,tgt_input['attention_mask']
        # video_feat,video_mask =visual_src_input[:,1:,:],visual_src_attention_mask[:,1:]
        # text_feat=self.relu(self.downsample(text_feat))
        # video_feat=self.relu(self.downsample(video_feat))
        # t_idx_token = torch.arange(text_feat.size(1))[None, :].repeat(text_feat.size(0), 1)
        # t_agg_weight = text_feat.new_ones(text_feat.size(0), text_feat.size(1), 1)
        # t_token_dict = {'x': text_feat,
        #                 'token_num': text_feat.size(1),
        #                 'idx_token': t_idx_token,
        #                 'agg_weight': t_agg_weight,
        #                 'mask': text_mask.detach()}
        # v_idx_token = torch.arange(video_feat.size(1))[None, :].repeat(video_feat.size(0), 1)
        # v_agg_weight = video_feat.new_ones(video_feat.size(0), video_feat.size(1), 1)
        # v_token_dict = {'x': video_feat,
        #                 'token_num': video_feat.size(1),
        #                 'idx_token': v_idx_token,
        #                 'agg_weight': v_agg_weight,
        #                 'mask': video_mask.detach()}
        # # action level
        # t_token_dict = self.t_block0(self.t_ctm0(t_token_dict))
        # v_token_dict = self.v_block0(self.v_ctm0(v_token_dict))
        # text_feat = t_token_dict["x"]
        # video_feat = v_token_dict["x"]
        # M_t2v_logits0, M_v2t_logits0, logits = self.action_level(text_feat, video_feat, text_mask, video_mask
        #                                                    )
        # M_t2v_logits0, M_v2t_logits0, logits = self.action_level(text_feat, video_feat, text_mask, video_mask,
        #                                                     t_token_dict, v_token_dict)
        # bt=text_feat.shape[0]
        # labels = torch.arange(bt, device=posterior_text_encoder_out.device, dtype=torch.long)
        # loss_struc = F.cross_entropy(M_t2v_logits0, labels)
        # loss_seq = F.cross_entropy(M_v2t_logits0, labels)
        # M_loss = (loss_struc + loss_seq) / 2
        # M_loss_t2v = self.loss_fct(M_t2v_logits0 * self.logit_scale.exp())
        # M_loss_v2t = self.loss_fct(M_v2t_logits0 * self.logit_scale.exp())
        # M_loss = (M_loss_t2v + M_loss_v2t) / 2
        # logits = torch.diagonal(logits, dim1=0, dim2=1).permute(2, 0, 1)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)#对图像和文本特征进行归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #pearson相关系数损失
        # pearson_los=pearson_correlation(image_features,text_features)*0.1
        #视觉之间与文本之间对应的相似度
        # si_cos=sim_cos(image_features,text_features,visual_src_attention_mask,tgt_input['attention_mask'])
        #不用全局特征的clip_loss
        # clip_los=clip_loss(visual_src_input[:,1:,:],self.encoder_hidden_states,self.logit_scale)

        #signcl损失
        # num_frames = visual_src_input.shape[1]-1
        # # text_length =tgt_input['attention_mask'].shape[1] # Assuming text_input is the corresponding text
        # # margin = min(20, max(10, int((num_frames // text_length+1) * 2.3))*2)
        # margin = max(2.5, int((num_frames// tgt_input['input_ids'].shape[1] + 1)/4 * 2.3)) * 2
        # num_negative = 7
        # margin = min(margin, int((num_frames - num_negative) / 2)) #ensure num_frames margin for negative sampling
        # cl_loss = self.cl_criterion(frames, margin=margin)


        # cosine similarity as logits，使用余弦相似度作为logits
        logit_scale = self.logit_scale.exp() 
        logits_per_image = logit_scale * image_features @ text_features.t() #计算图像特征和文本特征之间的点积，得到图像对文本的logits
        logits_per_text = logit_scale * text_features @ image_features.t() #计算文本特征和图像特征之间的点积，得到文本对图像的logits

        # video-sentence score TIB模块里的整体特征之间的对比损失,和上面的logits_per_text一样没啥区别
        # sentence_video_logits = logit_scale * torch.matmul(torch.matmul(text_features, self.global_mat_weight), image_features.t())
        # sim_loss=0.0

        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)
        #创建一个单位矩阵，用于作为真实标签
        #进行视频文本匹配任务
        # non-masked text and non-masked image 
        # vtm_loss, vtm_logits, vtm_labels = self.compute_vtm(text_embeds=text_embeds, 
        #                                                     text_atts=tgt_input['attention_mask'], 
        #                                                     image_embeds=visual_src_input, 
        #                                                     image_atts=visual_src_attention_mask, 
        #                                                     sim_i2t=logits_per_image.clone(), # for hard mining
        #                                                     sim_t2i=logits_per_text.clone()  # for hard mining
        #                                                    )

        #基于排序和基于Viterbi算法的细粒度对比学习损失
        # image_features1 = video_feat / (video_feat.norm(dim=-1, keepdim=True) + 1e-6)
        # text_features_phrase1 = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
        # logit_per_image1 = image_features1@text_features_phrase1.permute(0, 2, 1)
        # logit_per_text1 = text_features_phrase1@image_features1.permute(0, 2, 1)
        
        # preds_image_label = F.gumbel_softmax(logit_per_image1, dim=-1, tau=logit_scale, hard=True)
        # # gt_image_label=self.generate_gumbel_sort_gt(preds_label=preds_image_label)
        # gt_image_label=self.generate_gumbel_viterbi_gt(sim_matrices=logit_per_image1) #Viterbi算法
        # image_loss = F.cross_entropy(preds_image_label.permute(0, 2, 1), gt_image_label, reduction='mean')  # CE([b,t,n](one hot), [b,t](ground truth))
        # preds_text_label = F.gumbel_softmax(logit_per_text1, dim=-1, tau=logit_scale, hard=True)  # -> [b,n,t]
        # # gt_text_label =self.generate_gumbel_sort_gt(preds_label=preds_text_label)
        # gt_text_label=self.generate_gumbel_viterbi_gt(sim_matrices=logit_per_text1) #Viterbi算法
        # # text_loss = F.cross_entropy(preds_text_label.permute(0, 2, 1), gt_text_label, reduction='none')
        # # text_loss_masked = text_loss * tgt_input['attention_mask']
        # # loss = image_loss + text_loss_masked.mean()
        # text_loss = F.cross_entropy(preds_text_label.permute(0, 2, 1), gt_text_label, reduction='mean')
        # loss = (image_loss + text_loss)/2

        #Cico中实现的手语视频文本细粒度对比
        # I2T_sim,T2I_sim = self.flip_similarity_softmax(self.encoder_hidden_states, visual_src_input[:,1:,:], tgt_input['attention_mask'], visual_src_attention_mask[:,1:])
        # sim_loss=(self.con_loss(I2T_sim)+self.con_loss(T2I_sim))/2.0
        #置信度分数的对比模块
        img_embs,cap_embs=visual_src_input[:,1:,:],self.encoder_hidden_states
        cap_lens=tgt_input['attention_mask'].sum(dim=1)
        img_lens=visual_src_attention_mask[:,1:].sum(dim=1)
        # t_list = self.sim_embt(img_embs, cap_embs, cap_lens)
        v_list= self.sim_embv(img_embs, cap_embs, cap_lens,visual_src_attention_mask[:,1:])
        t_list= self.sim_embvt( cap_embs,img_embs, img_lens,tgt_input['attention_mask'])
        # sims = self.sim_enc(t_list, v_list)
        # sim_loss=self.con_loss(sims)
        I2T_sim,T2I_sim = self.flip_similarity_softmax(self.encoder_hidden_states, visual_src_input[:,1:,:], tgt_input['attention_mask'], visual_src_attention_mask[:,1:],v_list,t_list)
        return I2T_sim,T2I_sim , ground_truth

class FeatureExtracter(nn.Module):
    def __init__(self):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet() # InceptionI3d()，将预训练的2D CNN模型集成到CLIP中
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)#将预训练的1D CNN模型集成到CLIP中

    def forward(self,
                src: Tensor,
                src_length_batch
                ):
        src = self.conv_2d(src,src_length_batch)#src是输入的序列，src_length_batch 是输入序列的长度。
        src = self.conv_1d(src)

        return src

class Attention(nn.Module):#用于实现多头自注意力机制
    def __init__(self, dim, heads=16, dim_head=64, attn_drop=0.):
        super().__init__() #头部数量 heads、每个头部的维度 dim_head 和注意力丢弃率 attn_drop。
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5 #用于存储缩放因子，是 dim_head 的负半次幂
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)#定义了线性变换 self.to_qkv，用于将输入数据转换为查询、键和值。
        self.score = None
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads # b = batch size; n = sequence length; h = number of heads
        #将线性变换后的输出分为三个部分，分别作为查询、键和值
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)#将查询、键和值按头进行重新排列，将查询、键和值的维度从 b n (h d) 转换为 b h n d
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # 计算查询和键的点积，并乘以缩放因子。

        if mask is not None: #检查是否提供了注意力掩码
            mask = F.pad(mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]，将注意力掩码展平，并在开头添加一个元素，并将其设置为1，表示应该忽略的值
            mask = mask[:, None, None, :].float() #将填充后的掩码重新排列为 [b, 1, 1, h]，并将数据类型转换为浮点数
            dots -= 10000.0 * (1.0 - mask) #将注意力掩码与点积结果相减，忽略应该忽略的位置，
            #如果掩码值为 1，则点积结果不变；如果掩码值为 0，则点积结果变为负无穷大。这样可以确保在计算注意力权重时，不会被掩码值为 0 的位置影响
        attn = dots.softmax(dim=-1) #使用 softmax 函数计算注意力权重。将点积结果 dots 沿着最后一个维度进行 softmax 运算，得到注意力权重 attn
        self.score = attn 
        out = einsum('b h i j, b h j d -> b h i d', attn, v) #将注意力权重与值相乘，得到输出
        out = rearrange(out, 'b h n d -> b n (h d)') #将输出重新排列为 b n (h d) 的形状
        return out
    def visualize(self):
        return self.score #返回注意力权重
        
class FeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout) #定义一个 dropout 层，用于在训练过程中随机丢弃部分神经元

    def forward(self, x): #接受一个输入张量 x，形状为 [batch_size, sequence_length, dim]
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) #实例化一个 LayerNorm 层
        self.fn = fn #保存传递给 PreNorm 的模块

    def forward(self, x, **kwargs):#接受一个输入张量 x，形状为 [batch_size, sequence_length, dim]，并传递其他关键字参数
        return self.fn(self.norm(x), **kwargs)#首先使用 LayerNorm 层对输入张量 x 进行归一化，然后将归一化后的张量传递给函数 fn 进行处理

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):#attn_drop: 注意力Dropout的丢弃率，默认为 0。
        #proj_drop: 投影Dropout的丢弃率，默认为 0。
        super().__init__()
        num_heads = heads
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5 # 计算缩放因子，默认为头维度的负半次幂

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)#定义一个线性层，用于计算 Q 值
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.score = None #用于存储注意力得分的变量

    def forward(self, x, mask=None):#计算输入序列的交叉注意力表示

        B, N, C = x.shape #B 是批量大小，N 是序列长度，C 是输入维度
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]，将注意力掩码展平，并在开头添加一个元素，并将其设置为1，表示应该忽略的值
            mask = mask[:, None, None, :].float() #将填充后的掩码重新排列为 [b, 1, 1, h]，并将数据类型转换为浮点数
            attn -= 10000.0 * (1.0 - mask)
        attn = attn.softmax(dim=-1) #使用 softmax 函数计算注意力权重。将点积结果 dots 沿着最后一个维度进行 softmax 运算，得到注意力权重 attn
        attn = self.attn_drop(attn)
        self.score = attn

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        #通过注意力权重和值向量的矩阵乘法得到输出，然后重新塑形为 [B, 1, C] 形状
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x
    def visualize(self):
        return self.score

class Cross_att_layer(nn.Module):
    def __init__(self, dim=1024, heads=16, depth=2, dropout=0.1, attn_drop=0.0,  mlp_dim=768):
        #depth：网络的深度，默认为2。dropout：Dropout的比率，默认为0.1。attn_drop：注意力Dropout的比率，默认为0.0。mlp_dim：全连接层中的隐藏单元数量，默认为768。
        super(Cross_att_layer, self).__init__()

        self.layers = nn.ModuleList([])#初始化一个空的nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, attn_drop=attn_drop)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(dim, CrossAttention(dim, heads=heads, attn_drop=attn_drop)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])) # 为每个深度层添加一个包含四个模块的列表：自注意力、MLP、交叉注意力和MLP
        self.cls_token_f = nn.Parameter(torch.randn(1, 1, dim))#初始化一个可训练的参数，用于添加到图像特征向量的开头，作为分类头的输入
        self.cls_token_g = nn.Parameter(torch.randn(1, 1, dim))#初始化一个可训练的参数，用于添加到文本特征向量的开头，作为分类头的输入

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, inp_dim))

    def forward(self, f, fmask, g, gmask):
        B, N, C = f.shape
        cls_token_f = repeat(self.cls_token_f, '() n d -> b n d', b=B)#将 self.cls_token_f 重复 B 次，以匹配输入数据的批量大小
        f = torch.cat((cls_token_f, f), dim=1)#将分类cls标记添加到图像特征向量的开头
        cls_token_g = repeat(self.cls_token_g, '() n d -> b n d', b=B)
        g = torch.cat((cls_token_g, g), dim=1)
        for attn1, ff1, c_attn, ff3 in self.layers:
            f = attn1(f) + f
            f = ff1(f) + f

            g = attn1(g) + g
            g = ff1(g) + g

            f_g = c_attn(torch.cat((f[:, 0:1, :], g[:, 1:, :]), dim=1))#将图像特征与文本特征的剩余部分进行交叉注意,将 f 的第一个标记和 g 的第二个标记拼接起来，并通过跨注意力层 c_attn 处理
            g_f = c_attn(torch.cat((g[:, 0:1, :], f[:, 1:, :]), dim=1))#将文本特征与图像特征的剩余部分进行交叉注意,将 g 的第一个标记和 f 的第二个标记拼接起来，并通过跨注意力层 c_attn 处理
            f = torch.cat((g_f, f[:, 1:, :]), dim=1)#将 g_f 和 f 的剩余部分拼接起来。
            g = torch.cat((f_g, g[:, 1:, :]), dim=1)#将 f_g 和 g 的剩余部分拼接起来。
            f = ff3(f) + f
            g = ff3(g) + g

        return torch.cat((f[:, 0:1, :], g[:, 0:1, :]), dim=1)#返回由 f 和 g 的 CLS标记组成的输出序列

class V_encoder(nn.Module):
    def __init__(self,
                 emb_size,
                 feature_size,
                 config,
                 ):#emb_size：特征嵌入的维度，默认为1024。feature_size：输入特征的维度，默认为2048。
        super(V_encoder, self).__init__()
        
        self.config = config

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = [] #创建一个空列表 modules，用于存储网络模块。
        modules.append(nn.BatchNorm1d(emb_size)) #列表中添加一个批量归一化层，用于对 emb_size 大小的数据进行归一化。
        modules.append(nn.ReLU(inplace=True)) #向列表中添加一个 ReLU 激活层，用于对数据进行非线性变换。
        self.bn_ac = nn.Sequential(*modules) #创建一个顺序模型 self.bn_ac，按顺序包含批量归一化层和 ReLU 激活层

        for m in self.modules(): #遍历类的所有子模块。
            if isinstance(m, (nn.Conv1d,nn.Linear)): #判断子模块是否为线性变换层或卷积层。
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))#使用 Xavier 初始化方法对子模块的权重进行初始化，并根据 ReLU 激活函数的特性设置增益
            elif isinstance(m, nn.BatchNorm1d): 
                nn.init.constant_(m.weight, 1) #将批量归一化层的权重初始化为 1。
                nn.init.constant_(m.bias, 0) #将批量归一化层的偏置初始化为 0。
    
    def forward(self,
                src: Tensor,
                ):
      
        src = self.src_emb(src)#将输入数据传递给线性变换层 self.src_emb。
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1) #将数据传递给顺序模型 self.bn_ac，先进行批量归一化和 ReLU 激活，然后重新排列维度。

        return src

def config_decoder(config): #用于根据配置参数创建解码器模型
    decoder_type = _('decoder_type', 'LD', choices=['LD', 'LLMD']) #获取 decoder_type 参数的值，默认为 'LD'。
    config['model']["decoder_type"] = decoder_type
    if decoder_type == 'LD':#创建一个预训练的 MBartForConditionalGeneration 模型作为图像处理模型
        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'], ignore_mismatched_sizes = True, config = Path(config['model']['visual_encoder'])/'config.json')#ignore_mismatched_sizes 参数设置为 True，以忽略模型参数大小不匹配的问题
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes = True, config = Path(config['model']['transformer'])/'LLMD_config.json')
    
class gloss_free_model(nn.Module):
    def __init__(self, config, args, embed_dim=1024, pretrain=None):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args

        self.gloss_tokenizer = GlossTokenizer_S2G(config['model']['GlossTokenizer'])
        self.visual_head = VisualHead(cls_num=len(self.gloss_tokenizer))
        self.mapping = torch.nn.Sequential(
                torch.nn.Linear(in_features=512, out_features=1024),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=1024, out_features=1024)
            )

        # self.localattention=DeformableMultiHeadedAttention(
        #         query_type='not',
        #         query_nb=3,
        #         num_heads=8,
        #         size=512,
        #         dropout=0.5,
        #         num_keys=4,
        #     )
        # self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        # self.dropout = nn.Dropout(0.5)

        # self.backbone = FeatureExtracter() #创建一个 FeatureExtracter 对象，用于提取图像特征
        # self.mbart = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'])
        self.mbart = config_decoder(config)
        
 
        if config['model']['sign_proj']: #判断配置参数中是否启用了手语映射。
            self.sign_emb = V_encoder(emb_size=embed_dim,feature_size=embed_dim, config = config)#创建一个 V_encoder 对象，用于将手语映射到图像特征
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0 #设置 self.embed_scale 的值，默认为 1.0
        else:
            self.sign_emb = nn.Identity() #创建一个恒等映射，即不进行任何变换。
            self.embed_scale = 1.0
        
    def share_forward(self, src_input):#接受一个字典 src_input 作为输入，该字典包含了输入序列的 ID 和注意力掩码
        
        # frames_feature = self.backbone(src_input['input_ids'].cuda(), src_input['src_length_batch'])#将输入数据传递给 backbone 模型，得到图像特征
        # attention_mask = src_input['attention_mask'] #从输入字典中提取注意力掩码

        attention_mask = src_input['attention_mask']
        x=self.visual_head(src_input['input_ids'].cuda(),attention_mask.cuda())
        frames_feature=self.mapping(x["gloss_feature"])

        #local attention
        # x_norm = self.layer_norm(x["gloss_feature"]).cuda()
        # h= self.localattention(x_norm, x_norm, x_norm, attention_mask.cuda())
        # h = self.dropout(h) + x["gloss_feature"]
        # # x= self.feed_forward(h)
        # frames_feature=self.mapping(h)

        inputs_embeds = self.sign_emb(frames_feature) #将图像特征传递给 sign_emb 模型，得到输入序列的嵌入表示
        inputs_embeds = self.embed_scale * inputs_embeds #将输入序列的嵌入表示乘以 self.embed_scale

        return inputs_embeds, attention_mask

    def forward(self,src_input, tgt_input ):#接受一个源输入序列 src_input 和一个目标输入序列 tgt_input
        
        inputs_embeds, attention_mask = self.share_forward(src_input) #调用 share_forward 函数，传入 src_input，得到输入序列的嵌入表示和注意力掩码

        out = self.mbart(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),
                    # decoder_input_ids = tgt_input['input_ids'].cuda(),
                    labels = tgt_input['input_ids'].cuda(),
                    decoder_attention_mask = tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    ) #将输入序列的嵌入表示和注意力掩码传递给 MBartForConditionalGeneration 模型，并获取模型输出
        output = out['encoder_last_hidden_state'][:, 0, :] #从模型输出中提取最后一个编码器隐藏状态，并取第一维的所有值（即所有时间步）和第三维的所有值（即所有特征），并将结果赋值给 output
        return out['logits'], output #返回解码器的输出 logits 和编码器的最后一个隐藏状态。
    

    def generate(self,src_input,max_new_tokens,num_beams,decoder_start_token_id ):#接受一个源输入序列 src_input，最大新生成令牌数max_new_tokens，束宽num_beams 和 解码器起始令牌 IDdecoder_start_token_id
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),max_new_tokens=max_new_tokens,num_beams = num_beams,
                                decoder_start_token_id=decoder_start_token_id
                            )#调用解码器模型的 generate 方法，传入嵌入向量、注意力掩码、最大新生成令牌数、束宽和解码器起始令牌 ID，并将返回的结果赋值
        return out