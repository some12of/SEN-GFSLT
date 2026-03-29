from ctypes import util
from cv2 import IMREAD_GRAYSCALE
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
import numpy as np
import lmdb
import io
import time
from vidaug import augmentors as va
from augmentation import *

from loguru import logger

# global definition
from definition import *

from FeatureLoader import load_batch_feature
from collections import defaultdict  

class Normaliztion(object):#将图像归一化到[-1, 1]范围
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image): #是否是PIL.Image.Image类的实例
            Image = np.asarray(Image, dtype=np.uint8) #转换为NumPy数组，并设置数据类型为np.uint8
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class SomeOf(object): #随机选择多个数据增强方法
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5: #随机生成一个介于0和1之间的数，如果该数大于0.5，则应用transforms1变换
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip

class S2T_Dataset(Dataset.Dataset): #用于处理视频描述数据的加载和转换
    def __init__(self,path,path1,tokenizer,config,args,phase, training_refurbish=False):
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish # 用于训练过程中的refurbish重新整理
        
        self.raw_data = utils.load_dataset_file(path) #加载数据集文件，并将其赋值给类的属性self.raw_data。
        self.tokenizer = tokenizer 
        self.img_path = config['data']['img_path'] #配置文件中获取图片路径，并将其赋值给类的属性self.img_path
        self.phase = phase
        self.max_length = config['data']['max_length']
        

        annotation = utils.load_dataset_file(path1) #加载预训练的特征
        name2feature = {a['name']: a['sign'] for a in annotation}
        for i in range(len(annotation)):
            self.raw_data[annotation[i]["name"]]['head_rgb_input']=name2feature[annotation[i]["name"]]
        
        # raw_text=[]
        # for i in range(len(annotation)):
        #     raw_text.append(self.raw_data[annotation[i]["name"]]['text'])
        # with self.tokenizer.as_target_tokenizer(): # 使用self.tokenizer.as_target_tokenizer()方法将self.tokenizer转换为目标tokenizer
        #     tgt_text = self.tokenizer(raw_text, return_tensors="pt",padding = True,  truncation=True)
        # # 使用 defaultdict 来方便地统计词频  
        # word_freq = defaultdict(int)  
        # # 将张量转换为 Python 列表以便进行迭代  
        # input_ids_list = tgt_text['input_ids'].tolist()
        # # 遍历列表中的每个句子（每行）  
        # for sentence in input_ids_list:  
        #     # 遍历句子中的每个 ID  
        #     for id in sentence:  
        #         word_freq[id] += 1  
        # # 将 defaultdict 转换为普通字典（如果需要）  
        # word_freq_dict = dict(word_freq)
        # # 获取词表大小  
        # vocab_size = self.tokenizer.vocab_size

        self.list = [key for key,value in self.raw_data.items()] #获取数据集中的所有键，并将其赋值给类的属性self.list。
        # sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability，用于以50%的概率应用aug变换。
        # self.seq = va.Sequential([
        #     # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
        #     # va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
        #     sometimes(va.RandomRotate(30)), # 随机旋转the video with a degree randomly choosen from [-30, 30]
        #     sometimes(va.RandomResize(0.2)),#这个操作会在50%的情况下对图像进行随机缩放，缩放的比例在原始尺寸的80%到120%之间。
        #     # va.RandomCrop(size=(256, 256)),
        #     sometimes(va.RandomTranslate(x=10, y=10)),#这个操作会对图像进行随机平移，平移的距离在x轴和y轴方向上都不超过10像素

        #     # sometimes(Brightness(min=0.1, max=1.5)),
        #     # sometimes(Contrast(min=0.1, max=2.0)),

        # ])
        # self.seq_color = va.Sequential([
        #     sometimes(Brightness(min=0.1, max=1.5)), #在50%的情况下对图像的亮度进行随机调整，调整的范围在原始亮度的10%到150%之间。
        #     sometimes(Color(min=0.1, max=1.5)), #在50%的情况下对图像的颜色进行随机调整，调整的范围在原始颜色的10%到150%之间。
        #     # sometimes(Contrast(min=0.1, max=2.0)),
        #     # sometimes(Sharpness(min=0.1, max=2.))
        # ])
        # self.seq = SomeOf(self.seq_geo, self.seq_color)
 
        


    def __len__(self):
        return len(self.raw_data) #__len__方法将返回self.raw_data的长度，即数据集中的原始数据数量。
    
    def __getitem__(self, index):
        key = self.list[index] # 索引index获取对应的键值key，这个key用来从原始数据中获取对应的样本
        sample = self.raw_data[key] # 获取数据集中的第index条数据，并将其赋值给变量sample
        tgt_sample = sample['text'] #提取目标文本数据。
        length = sample['length'] #提取长度
        
        name_sample = sample['name'] #提取视频的名字

        # img_sample = self.load_imgs([self.img_path + x for x in sample['imgs_path']]) # 加载图像数据，并将其赋值给变量img_sample
        img_sample=sample['head_rgb_input']
        
        return name_sample,img_sample,tgt_sample,index
    


    def load_imgs(self, paths):

        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ]) #将图像转换为张量（tensor），对图像进行归一化，使得图像数据的均值为[0.485, 0.456, 0.406]，标准差为[0.229, 0.224, 0.225]。
        if len(paths) > self.max_length: #检查paths（图像路径的列表）的长度是否超过了self.max_length，如果超过了，则随机选择self.max_length个路径。
            tmp = sorted(random.sample(range(len(paths)), k=self.max_length))
            new_paths = []
            for i in tmp:
                new_paths.append(paths[i])
            paths = new_paths
    
        imgs = torch.zeros(len(paths),3, self.args.input_size,self.args.input_size) # 初始化一个长度为len(paths)的张量为imgs，该张量的形状为（3，224，224）。
        crop_rect, resize = utils.data_augmentation(resize=(self.args.resize, self.args.resize), crop_size=self.args.input_size, is_train=(self.phase=='train'))
        #使用数据增强方法对图像进行裁剪和缩放。
        batch_image = [] # 初始化一个空列表batch_image，用于存储预处理后的图像。
        # def cv_imread(file_path):
        #     cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
        #     return cv_img
        for i,img_path in enumerate(paths):#遍历paths列表中的每个图像路径，读取对应的图像文件，并将其从BGR格式转换为RGB格式。
            img = cv2.imread(img_path) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img) #将图像从numpy数组转换为PIL.Image.Image类的实例。
            batch_image.append(img) #将处理后的图像添加到batch_image列表中。

        if self.phase == 'train': # 如果当前是训练阶段，则使用数据增强方法对图像进行处理。
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):#将预处理后的图像添加到imgs张量中。
            img = img.resize(resize) #将图像缩放到指定大小。
            img = data_transform(img).unsqueeze(0) # 图像进行归一化操作，并添加一个额外的维度，以便将图像数据转换为PyTorch所需的格式。  
            imgs[i,:,:,:] = img[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]] #裁剪后的图像数据。crop_rect是一个元组，表示裁剪矩形的四个坐标值，通常用于将图像裁剪为指定大小。
        
        return imgs

    def collate_fn(self,batch): # 用于合并批次中样本的函数，将数据批中的所有样本打包成一个大的数据张量，以便在训练过程中对模型进行训练
        
        tgt_batch,img_tmp,src_length_batch,name_batch,index_batch = [],[],[],[],[] #初始化四个列表：tgt_batch（用于存储目标数据）、img_tmp（用于存储图像数据）、src_length_batch（用于存储每个样本的目标长度）和name_batch（用于存储样本名称）。

        for name_sample, img_sample, tgt_sample,idx_sample in batch: #遍历批次中的每个样本，并将它们添加到相应的列表中。

            name_batch.append(name_sample)

            img_tmp.append(img_sample)

            tgt_batch.append(tgt_sample)

            index_batch.append(idx_sample)
        # max_len = max([len(vid) for vid in img_tmp]) #计算数据批中最长样本的长度max_len
        # video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in img_tmp]) #计算数据批中每个样本需要填充到的长度，使样本的长度为4的整数倍，并加上16
        # left_pad = 8 #left_pad是每个样本左侧的填充长度，right_pad是每个样本右侧的填充长度
        # right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8 # 计算每个样本右侧的填充长度，使其长度为4的整数倍
        # max_len = max_len + left_pad + right_pad # 计算填充后的最大长度max_len
        # padded_video = [torch.cat(
        #     (
        #         vid[0][None].expand(left_pad, -1, -1, -1),
        #         vid,
        #         vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
        #     )
        #     , dim=0)
        #     for vid in img_tmp] #使用torch.cat函数将这些帧连接在一起，形成一个填充过的视频样本。torch.cat函数用于将多个张量连接在一起，形成一个新的张量。
        
        # img_tmp = [padded_video[i][0:video_length[i],:,:,:] for i in range(len(padded_video))] #该列表包含所有填充过的视频样本，长度为video_length
        
        for i in range(len(img_tmp)): #计算img_tmp列表中所有视频样本的长度，并将这些长度添加到src_length_batch列表中
            src_length_batch.append(len(img_tmp[i]))
        src_length_batch = torch.tensor(src_length_batch) #将src_length_batch列表转换为一个PyTorch张量，以便将其用作模型的输入
        
        # img_batch = torch.cat(img_tmp,0) #将img_tmp列表中的所有视频样本连接成一个大的PyTorch张量，以便将其用作模型的输入

        # new_src_lengths = (((src_length_batch-5+1) / 2)-5+1)/2 # 计算新长度，即将原始长度除以2，并减去5，然后加上5，最后除以2。
        # new_src_lengths = new_src_lengths.long() # 将新长度转换为整数，这个新的源长度将用于生成遮罩
        # mask_gen = [] # 初始化一个空列表mask_gen，用于存储生成的遮罩
        # for i in new_src_lengths: #遍历新的源长度new_src_lengths，并为每个长度生成一个遮罩。遮罩是一个PyTorch张量，其中所有元素都为1，我们加上7，然后将其添加到mask_gen列表中
        #     tmp = torch.ones([i]) + 7
        #     mask_gen.append(tmp)
        # mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX,batch_first=True) # 使用pad_sequence函数将mask_gen列表中的所有元素连接成一个大的PyTorch张量，并将PAD_IDX作为填充值
        # img_padding_mask = (mask_gen != PAD_IDX).long() # 使用mask_gen生成一个图像填充掩码，该掩码具有与img_batch相同的形状，其中所有元素都为0或1，0表示填充值，1表示非填充值
        with self.tokenizer.as_target_tokenizer(): # 使用self.tokenizer.as_target_tokenizer()方法将self.tokenizer转换为目标tokenizer
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt",padding = True,  truncation=True) #使用self.tokenizer来将目标批次tgt_batch转换为模型输入。我们使用return_tensors="pt"来返回PyTorch张量，并启用填充和截断

        

        src_input = {}
        # src_input['input_ids'] = img_batch #作为input_ids键的值。img_batch是一个PyTorch张量，包含所有填充过的视频样本。
        # src_input['attention_mask'] = img_padding_mask #img_padding_mask是一个布尔张量，表示每个位置是否为填充位置。
        src_input['input_ids'],src_input['attention_mask'], src_input['new_src_length_batch'] = load_batch_feature(features=[i + 1.0e-8 for i in img_tmp])
        src_input['src_length_batch'] = src_length_batch 
        # src_input['new_src_length_batch'] = new_src_lengths # 将新的源长度new_src_lengths添加到源输入字典中
        src_input['sample_idx']=index_batch 

        if self.training_refurbish:
            masked_tgt = utils.NoiseInjecting(tgt_batch, self.args.noise_rate, noise_type=self.args.noise_type, random_shuffle=self.args.random_shuffle, is_train=(self.phase=='train'))
            #utils.NoiseInjecting向目标批次tgt_batch注入噪声。utils.NoiseInjecting是一个函数，它将噪声注入到目标批次中，以模拟实际应用中的噪声情况
            with self.tokenizer.as_target_tokenizer(): #将self.tokenizer设置为目标tokenizer。这是因为self.tokenizer默认是源tokenizer，我们需要将其设置为目标tokenizer，以便正确地将目标批次转换为模型输入
                masked_tgt_input = self.tokenizer(masked_tgt, return_tensors="pt", padding = True,  truncation=True)
            return src_input, tgt_input, masked_tgt_input
        return src_input, tgt_input

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.' #我们尝试打印Dataset对象时，__str__方法将返回一个包含有关数据集信息的字符串。






