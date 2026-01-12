import torch
import torch.nn as nn
from metric import ndcg
from config import hparams
from model.NRMS import NRMS
import pickle
import numpy as np
from dataset import ValidDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

news_file=hparams['news_file']
behaviors_file=hparams['behaviors_file']
w2v_file=hparams['w2v_file']

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(w2v_file, 'rb') as f:
    data = pickle.load(f)
    w2id = data['w2id']
    matrix = torch.from_numpy(data['embedding']).float() # 这就是从 GloVe 提取的矩阵

# 加载模型
model = NRMS(hparams,matrix)
model_path=hparams['model_path']
state_dict=torch.load(model_path,map_location=device)
model.load_state_dict(state_dict)
model.to(device)

dataset=ValidDataset(news_file, behaviors_file, w2v_file,max_len=20,max_hist_len=50)
valid_loader = DataLoader(dataset, batch_size=1, shuffle=False)

def evaluate(model):
    model.eval()
    res=[]
    with torch.no_grad():
        for click_docs,cand_docs,labels in tqdm(valid_loader):
            click_docs=click_docs.to(device) # 【1,max_hist_len,max_len】
            cand_docs=cand_docs.to(device) # [1,num_cands,max_len]
            labels=labels.to(device).squeeze(0) # [num_cands]

            logits=model(click_docs,cand_docs) # [1, num_cands]
            logits=logits.squeeze(0)
            if labels.sum()>0 and labels.sum()<len(labels):
                scores=ndcg(logits,labels,k=10)
                res.append(scores)
    mean_ndcg=np.mean(res)
    print(f"\n评估完成! nDCG@10: {mean_ndcg:.4f}")

evaluate(model)