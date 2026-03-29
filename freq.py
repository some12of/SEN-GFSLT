import torch
from collections import defaultdict, Counter  
import utils as utils
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig 
import pickle  
import json  
from statistics import median  
# import matplotlib 
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  
import numpy as np
import scipy.io as scio
from scipy.stats import norm  
import matplotlib
import scipy
import scipy.stats
from matplotlib import rcParams
#rcParams['font.family']='sans-serif'
# rcParams['font.family']='sans-serif'
# rcParams['font.sans-serif']=['Arial']

tokenizer = MBartTokenizer.from_pretrained('./pretrain_models/MBart_trimmed')

# 定义一个函数来统计词频  
def get_word_freq(texts, tokenizer):
    with tokenizer.as_target_tokenizer(): # 使用self.tokenizer.as_target_tokenizer()方法将self.tokenizer转换为目标tokenizer 
        input_ids_list = [torch.tensor(tokenizer(text, return_tensors="pt", padding=True, truncation=True)['input_ids']).squeeze().tolist() for text in texts]  
    word_freq = Counter()
    # print(input_ids_list)  
    for sentence in input_ids_list:  
        word_freq.update(sentence)  
    return dict(word_freq)
  
def get_freq(texts, tokenizer=None):   
    # 获取分词器的词汇表  
    # vocab = tokenizer.vocab  
    # 初始化一个新的计数器来统计词汇频  
    word_freq1 = Counter()  
    # # 将ID频转换为词汇频  
    # for id, freq in id_freq.items():  
    #     # 检查ID是否在词汇表中（有些ID可能是填充或特殊标记）  
    #     if id in vocab:  
    #         word = vocab[id]  
    #         word_freq[word] += freq  
    # # 返回以词汇为键的词频字典
    for text in texts:  
        # 假设文本中的词已经由空格分隔开  
        words = text.split()  # 将文本按空格分割成单词列表  
        word_freq1.update(words)  # 更新词频计数器 
    return dict(word_freq1) 

def get_raw_text(path,path1):
    raw_data = utils.load_dataset_file(path)
    annotation = utils.load_dataset_file(path1) #加载预训练的特征
    raw_text=[]
    for i in range(len(annotation)):
        raw_text.append(raw_data[annotation[i]["name"]]['text'])
    return raw_text
train_texts=get_raw_text('./data/Phonexi-2014T/labels.train','/home/zfc2b/CV-SLT-main/experiments/configs/SingleStream/phoenix-2014t_s2g/extract_features/head_rgb_input/train.pkl')
val_texts=get_raw_text('./data/Phonexi-2014T/labels.dev','/home/zfc2b/CV-SLT-main/experiments/configs/SingleStream/phoenix-2014t_s2g/extract_features/head_rgb_input/dev.pkl')
test_texts=get_raw_text('./data/Phonexi-2014T/labels.test','/home/zfc2b/CV-SLT-main/experiments/configs/SingleStream/phoenix-2014t_s2g/extract_features/head_rgb_input/test.pkl')

train_word_freq= get_word_freq(train_texts, tokenizer)
# 获取验证集的词频字典  
val_word_freq = get_word_freq(val_texts, tokenizer)

# train_freq=get_freq(train_texts)
# print(train_freq)   
# 获取测试集的词频字典  
# test_word_freq = get_word_freq(test_texts, tokenizer)  
#  + Counter(test_word_freq)  
# 合并三个字典的词频  
combined_word_freq = Counter(train_word_freq) + Counter(val_word_freq)
# 要更新的键值对列表  
# updates = {  
#     1: 0,  
#     2: 0,
#     2430: 0 
# }  
# # 遍历更新列表并修改字典中的值  
# for key, value in updates.items():  
#     if key in combined_word_freq:  
#         combined_word_freq[key] = value 
# combined_word_freq[1] = 5000
# 转换为普通字典（如果需要）并排序  
sorted_word_freq = dict(sorted(combined_word_freq.items(), key=lambda item: item[1]))
# 计算频数的中位数  
freq_values = list(sorted_word_freq.values())  
median_freq = median(freq_values)
mean_freq = np.mean(freq_values)  
std_dev_freq = np.std(freq_values)
mad = np.median(np.abs(np.array(freq_values) - median_freq))  
# 使用1.4826因子将MAD转换为类似于标准差的量度（在正态分布下）  
sigma_mad = mad * 1.4826
print(freq_values)
print(median_freq,sigma_mad,mean_freq,std_dev_freq)

# 提取词频值  
word_freqs = list(sorted_word_freq.values())

loss_weight = [torch.pow(torch.tensor(item) / 10, torch.tensor(2)) * torch.exp(-1 * 1.75 * torch.tensor(item) / 10)*torch.exp(-0.5 *  torch.pow(torch.tensor(item-90), torch.tensor(2))/torch.pow(torch.tensor(370.0), torch.tensor(2))) for item in word_freqs]
b = max(loss_weight)
loss_weight = torch.tensor(torch.tensor(loss_weight) / b * (np.e - 1) + 1)

loss_weight1= [torch.pow(torch.tensor(item) / 10, torch.tensor(2)) * torch.exp(-1 * 1.75 * torch.tensor(item) / 10) for item in word_freqs]
b1 = max(loss_weight1)
loss_weight1 = torch.tensor(torch.tensor(loss_weight1) / b1* (np.e - 1) + 1).numpy()

loss_weight2= [torch.exp(-0.5 *  torch.pow(torch.tensor(item-20), torch.tensor(2))/torch.pow(torch.tensor(10.0), torch.tensor(2))) for item in word_freqs]
b2 = max(loss_weight2)
loss_weight2 = torch.tensor(torch.tensor(loss_weight2) / b2* (np.e - 1) + 1).numpy()
loss_weight3=(loss_weight2+loss_weight1)/2
# sorted_word_freqs1 = np.sort(loss_weight) 
# log_freqs1 = np.log1p(sorted_word_freqs1)
# 对词频进行排序  
sorted_word_freqs = np.sort(word_freqs) 
log_freqs = np.log1p(sorted_word_freqs) 
# 生成排序后的索引（从0开始）  

sorted_indices = np.arange(len(sorted_word_freqs))  
# 绘制词频排序图  
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(loss_weight)), loss_weight, marker='o', linestyle='-', color='r')
# plt.plot(range(len(loss_weight1)), loss_weight1, marker='o', linestyle='dotted', color='b')   
# plt.plot(range(len(loss_weight2)), loss_weight2, marker='o', linestyle='dotted', color='g')
# plt.plot(range(len(loss_weight3)), loss_weight3, marker='o', linestyle='dotted', color='pink')       
# # plt.bar(range(len(log_freqs)), log_freqs, color='skyblue')  # 使用条形图展示词频 
# # plt.plot(range(len(log_freqs)), log_freqs, marker='o', linestyle='-', color='b')  
# #添加标题和标签  
# # plt.title('词频排序分布图')  
# # plt.xlabel('排序后的索引（从低频到高频）')  
# # plt.ylabel('词频数')  
# # 显示网格线（可选）  
# plt.grid(True) 
# # 显示图形  
# plt.xticks(rotation=45)  # 旋转x轴标签，以便更好地显示  
# plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域  
# plt.show()


#调整局部帧的注意权重
# 参数设置
num_frames = 7  # 总帧数（包括自身）
query_index = 3  # 查询帧的索引（假设索引从0开始，且中间帧为第3帧）
peak_value = 0.56  # 中间帧的注意权重最高值
std_dev = 1.0 # 标准差，控制曲线的平滑度和宽度
mean=3
# 生成高斯分布的注意力权重
x1=np.arange(num_frames)
x= np.linspace(0, num_frames, 70)  # x轴上的位置
#局部注意产生的权重
data=[0.0,0.1,0.38,0.56,0.29,0.2,0.0]

# 使用scipy的norm.pdf函数计算高斯分布的注意力权重
# 注意：norm.pdf计算的是概率密度函数值，因此需要对结果进行缩放以确保中间值为peak_value
gaussian_weights = peak_value * norm.pdf(x, mean, std_dev) / np.max(norm.pdf(x, mean, std_dev))
gaussian_weights1 = peak_value * norm.pdf(x1, mean, std_dev) / np.max(norm.pdf(x1, mean, std_dev))
# gaussian_weights1 = np.exp(-0.5 * (x1 - query_index) ** 2 / sigma ** 2)
# gaussian_weights = np.exp(-0.5 * (x - query_index) ** 2 / sigma ** 2)
# gaussian_weights /= np.sum(gaussian_weights)  # 归一化，使权重之和为1
# 打印权重
datax=(data*gaussian_weights1+data)/1.65
print("Attention weights:", gaussian_weights)
# 绘制高斯分布图
plt.figure(figsize=(10, 5))
plt.plot(x, gaussian_weights, linestyle='-', color='blue', label='Gaussian Distribution')
plt.scatter(x1, data, linestyle='-', color='grey')
plt.scatter(x1, datax, linestyle='-', color='green')
# 标记注意权重的位置
# for i, weight in enumerate(gaussian_weights):
#     plt.scatter(x[i], weight, color='red', zorder=5)  # 使用红色点标记权重位置
#     plt.text(x[i], weight + 0.01, f'{weight:.4f}', ha='center', va='bottom', fontsize=10, color='green')  # 在点上方显示权重值
# 标记七个位置的注意权重
for i, weight in enumerate(gaussian_weights1):
    plt.scatter(x1[i], weight, color='red', zorder=5)  # 使用红色点标记权重位置
    # 在点旁边显示权重值，避免重叠并调整文本位置
    y_text = weight + 0.01 * (max(gaussian_weights) - min(gaussian_weights)) if weight != max(gaussian_weights) else weight + 0.005
    plt.text(x1[i], y_text, f'{weight:.4f}', ha='center', va='bottom', fontsize=10, color='green')

plt.title('Gaussian Attention Weights')
plt.xlabel('Frame Index')
plt.ylabel('Attention Weight')
plt.grid(True)
plt.legend()
plt.show()

# 将字典保存为 pickle 文件  
# with open('word_freq_dict.pkl', 'wb') as f:  
#     pickle.dump(sorted_word_freq, f) 
# print(f"Median frequency: {median_freq}") 
# print(sorted_word_freq) 