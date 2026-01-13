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
    def forward(self,context,mask=None):
        """
        加性注意力机制
        :param context: [batch_size,seq_len,in_dim]
        :return: outputs [batch_size，seq_len,out_dim],weights[batch_size,seq_len]
        """

        # proj->[B,seq_len,v_size], proj_v->[B,seq_len,1]
        # 因为下一步要用softmax打分得到每个token的权重，需要去除最后一维度
        weights=self.proj_v(self.proj(context)).squeeze(-1)
        if mask is not None:
            weights=weights.masked_fill(mask,-65500.0)
        weights=torch.softmax(weights,dim=-1) # [B,seq_len]
        # bmm批量矩阵乘法，要求两个输入都必须是3D张量，且第一维相等
        # unsqueeze升维 weights->[B,1,seq_len]，最终得到[B,seq_len]
        return torch.bmm(weights.unsqueeze(1),context).squeeze(1),weights


class TextEncoder(nn.Module):
    def __init__(self, hparams,weight=None):
        super(TextEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(hparams['vocab_size'], hparams['embed_dim'], padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False,padding_idx=0)

        self.input_projection=nn.Linear(hparams['embed_dim'],hparams['encoder_size'])

        # 2. 多头自注意力层,embed_dim指定模型的输入维度，输出维度默认等于输入维度
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.hparams['encoder_size'],
            num_heads=hparams['nhead'],
            dropout=0.1,
            batch_first=True
        )
        # 3. 注意力池化层
        self.additive_attention = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])

    def forward(self, x):
        """
        将新闻中各个token embedding聚合为一个embedding
        :param x: [batch_size, seq_len], 划分后的新闻标题数据，固定长度为seq_len
        :return: [batch_size,encoder_size]
        """
        padding_mask=(x==0)
        padding_mask[padding_mask.all(dim=1), 0] = False

        x = F.dropout(self.embedding(x), p=0.2, training=self.training) # [B,seq_len,embed_dim]
        x = self.input_projection(x)  # [B, seq_len, encoder_dim]

        output,_=self.multihead_attention(x,x,x,key_padding_mask=padding_mask) # [B,seq_len,embed_dim]

        output = F.dropout(output, p=0.2, training=self.training)

        output,_=self.additive_attention(output,mask=padding_mask) #　[B,encoder_size]
        return output
