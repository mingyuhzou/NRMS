import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple,Optional
class AdditiveAttention(nn.Module):
    # 加性注意力池化，将N个词向量聚合为一个代表整个文章的向量
    def __init__(self,in_dim,v_size):
        super(AdditiveAttention,self).__init__()

        self.in_dim=in_dim
        self.v_size=v_size

        self.proj=nn.Sequential(nn.Linear(self.in_dim,self.v_size),nn.Tanh())
        self.proj_v=nn.Linear(self.v_size,1)
    def forward(self,context):
        """
        加性注意力机制
        :param context: [batch_size,seq_len,in_dim]
        :return: outputs [batch_size，seq_len,out_dim],weights[batch_size,seq_len]
        """

        # proj->[B,seq_len,v_size], proj_v->[B,seq_len,1]
        # 因为下一步要用softmax打分得到每个token的权重，需要去除最后一维度
        weights=self.proj_v(self.proj(context)).squeeze(-1)
        weights=torch.softmax(weights,dim=-1) # [B,seq_len]
        # bmm批量矩阵乘法，要求两个输入都必须是3D张量，且第一维相等
        # unsqueeze升维 weights->[B,1,seq_len]，最终得到[B,seq_len]
        return torch.bmm(weights.unsqueeze(1),context).squeeze(1),weights


class TextEncoder(nn.Module):
    def __init__(self, hparams,weight=None):
        super(TextEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding=nn.Embedding(100,300)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False,padding_idx=0)

        # 2. 多头自注意力层,embed_dim指定模型的输入维度，输出维度默认等于输入维度
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.hparams['embed_dim'],
            num_heads=hparams['nhead'],
            dropout=0.1,
            batch_first=True
        )
        # 全连接层，使得新闻向量的维度不等于词向量
        self.proj=nn.Linear(self.hparams['embed_dim'],hparams['encoder_size'])
        # 3. 注意力池化层
        self.additive_attention = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])

    def forward(self, x):
        # x [batch_size, seq_len], 划分后的新闻标题数据，固定长度为seq_len
        x = F.dropout(self.embedding(x), p=0.2, training=self.training)
        output,_=self.multihead_attention(x,x,x)
        output = F.dropout(output, p=0.2, training=self.training)
        output=self.proj(output)
        output,_=self.additive_attention(output)
        return output
