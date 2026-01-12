# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import polars as pl
import pickle
import random
import re
from processData import processNews
# %%
class trainDataset(Dataset):
    def __init__(self,news_file,behaviors_file,w2v_file,max_len,max_hist_len,neg_num):
        self.news_dict = processNews(news_file,w2v_file,max_len)

        self.behaviors=pl.read_parquet(behaviors_file)
        self.max_len=max_len
        self.max_hist_len=max_hist_len
        self.neg_num=neg_num

        with open(w2v_file,'rb') as f:
            self.w2id=pickle.load(f)['w2id']

    # def sent2id(self,tokens):
    #     """
    #     将token转换为对应embedding的idx
    #     """
    #
    #     clean_tokens=[]
    #     # 清洗标点符号，变转换为小写
    #     for t in tokens:
    #         t_clean=re.sub(r'[^\w\s]', '', t).lower().strip()
    #         if t_clean:
    #             clean_tokens.append(t_clean)
    #     # 在embedding中找到对应idx，如果没有则映射为UNK
    #     token_ids=[self.w2id.get(t,1) for t in clean_tokens]
    #
    #     # 补齐为max_len
    #     token_ids+=[0]*(self.max_len-len(token_ids))
    #     # 截断
    #     return token_ids[:self.max_len]

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row=self.behaviors.row(idx,named=True)

        history=row['history']
        # 用户点击长度只保留最近的
        if len(history)>self.max_hist_len:
            history=history[-self.max_hist_len:]

        # 转换
        click_docs=[self.news_dict.get(nid,self.news_dict['<PAD>']) for nid in history]
        # 补齐历史长度
        pad_vec = self.news_dict['<PAD>']
        while len(click_docs) < self.max_hist_len:
            click_docs.append(pad_vec)
        
        impressions=row['impressions']
        pos_cands=[i.split('-')[0] for i in impressions if i.split('-')[1]=='1']
        neg_cands=[i.split('-')[0] for i in impressions if i.split('-')[1]=='0']
        
        # 随机抽取一个正样本
        target_pos=random.choice(pos_cands)
        # 负样本采样
        if len(neg_cands)>self.neg_num:
            target_negs=random.sample(neg_cands,self.neg_num)
        else:
            # 不够则重复采样
            target_negs=random.choices(neg_cands,k=self.neg_num) if neg_cands else ['<PAD_NID>'] * self.neg_num
        
        # 训练用的候选集为1个正样本+K个负样本
        cand_nids=[target_pos]+target_negs
        # 转换
        cand_docs = [self.news_dict.get(nid, self.news_dict['<PAD>']) for nid in cand_nids]
        
        label=0
        return (
            torch.LongTensor(click_docs), # [max_hist_len,max_len] 
            torch.LongTensor(cand_docs), # 【num_hand,max_len】
            torch.tensor(label),
        )

class ValidDataset(Dataset):
    def __init__(self,news_file,behaviors_file,w2v_file,max_len,max_hist_len):
        news_df=pl.read_parquet(news_file)
        self.news_dict = {
            row['news_id']: re.findall(r'\w+', row['title'].lower())  # 预处理时就转小写去符号
            for row in news_df.iter_rows(named=True)
        }

        self.behaviors=pl.read_parquet(behaviors_file)
        self.max_len=max_len
        self.max_hist_len=max_hist_len

        with open(w2v_file,'rb') as f:
            self.w2id=pickle.load(f)['w2id']

    def sent2id(self, tokens):
        """
        将token转换为对应embedding的idx
        """

        clean_tokens = []
        # 清洗标点符号，变转换为小写
        for t in tokens:
            t_clean = re.sub(r'[^\w\s]', '', t).lower().strip()
            if t_clean:
                clean_tokens.append(t_clean)
        # 在embedding中找到对应idx，如果没有则映射为UNK
        token_ids = [self.w2id.get(t, 1) for t in clean_tokens]

        # 补齐为max_len
        token_ids += [0] * (self.max_len - len(token_ids))
        # 截断
        return token_ids[:self.max_len]

    def __getitem__(self,idx):
        row=self.behaviors.row(idx,named=True)
        history=row['history']
        if len(history)>self.max_hist_len:
            history=history[-self.max_hist_len:]
        click_docs = []
        for nid in history:
            title = self.news_dict.get(nid, []) # 转换为title文本
            click_docs.append(self.sent2id(title)) # 转换为对应embedding的id

        while len(click_docs) < self.max_hist_len:
            click_docs.append([0] * self.max_len)
        
        # 无需采样
        impressions=row['impressions']
        cand_nids=[i.split('-')[0] for i in impressions]
        labels=[int(i.split('-')[1]) for i in impressions]

        cand_docs=[self.sent2id(self.news_dict.get(nid,[])) for nid in cand_nids]
        
        return (
            torch.LongTensor(click_docs), torch.LongTensor(cand_docs),torch.FloatTensor(labels),
        )
    def __len__(self):
        return len(self.behaviors)