# global definition
UNK_TOKEN = "<unk>" #表示未知的单词。如果在训练数据中出现了模型没有见过的单词，那么这个单词会被替换为UNK_TOKEN。
PAD_TOKEN = "<pad>" #用于表示文本序列的填充。当不同长度的文本序列需要被输入到模型中时，较短的序列会被填充到和最长序列相同的长度。
BOS_TOKEN = "<bos>" 
EOS_TOKEN = "<eos>"
SI_TOKEN = "<si>" #这是一个特殊的标记，用于表示句子的开始。
SI_IDX,PAD_IDX,UNK_IDX,BOS_IDX, EOS_IDX = 0,1,2,3,4 #这些是上述特殊标记在词汇表中的索引
SPECIAL_SYMBOLS = ['<si>','<pad>', '<unk>','<bos>', '<eos>'] 
WORD_MASK = "<mask>" #用于在预训练阶段遮挡单词，以便让模型预测被遮挡的单词。
