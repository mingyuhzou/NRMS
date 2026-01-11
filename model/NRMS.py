import torch
import torch.nn as nn
from TextEncoder import TextEncoder,AdditiveAttention
import torch.nn.functional as F

class NRMS(nn.Module):
    def __init__(self,hparams,weigth=None):
        super(NRMS, self).__init__()
        self.hparams = hparams
        self.doc_encoder=TextEncoder(hparams,weight=weigth)
        self.mha=nn.MultiheadAttention(hparams['encoder_size'],hparams['nhead'],dropout=0.1,batch_first=True)
        self.additive_attn=AdditiveAttention(hparams['encoder_size'],hparams['v_size'])
        self.criterion=nn.CrossEntropyLoss()
    def forward(self,clicks,cands,labels=None):
        """
        :param clicks: [num_user,num_click_docs(历史长度),seq_len]
        :param cands:  [num_user,num_cand_docs(候选数量),seq_len]
        :param labels:
        :return:
        """
        num_click_docs=clicks.shape[1]
        num_cand_docs=cands.shape[1]
        num_user=clicks.shape[0]
        seq_len=clicks.shape[2]

        # 把所有新闻看作一个大批次，DocEncoder只能处理二维输入
        clicks=clicks.reshape(-1,seq_len)
        cands=cands.reshape(-1,seq_len)

        click_emb=self.doc_encoder(clicks)
        cand_emb=self.doc_encoder(cands)

        # 转换回来
        click_emb=click_emb.reshape(num_user,num_click_docs,-1)
        cand_emb=cand_emb.reshape(num_user,num_cand_docs,-1)

        # 对用户兴趣建模
        click_output,_=self.mha(click_emb,cand_emb,click_emb)
        click_output=F.dropout(click_output,p=0.2,training=self.training)
        click_repr,_=self.additive_attn(click_output)

        # 点击预测
        logits=torch.bmm(click_repr.unsequeeze(1),cand_emb).squeeze(1)
        if labels is not None:
            loss=self.criterion(logits,labels)
            return loss,logits
        return torch.sigmoid(logits)