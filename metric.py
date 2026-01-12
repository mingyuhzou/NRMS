import torch


def ndcg(probs, labels, k):
    # probs: [num_cands], labels: [num_cands]
    num_cands=probs.shape[0]
    k=min(num_cands,k)
    _, indices = torch.topk(probs, k)
    relevant = labels[indices]  # 取出前k个对应的真实标签

    # 计算 DCG: sum(rel / log2(pos + 1))
    # 在推荐中，通常 rel 是 0 或 1
    rank_alpha = torch.arange(2, k + 2, device=probs.device).float()
    dcg = torch.sum(relevant / torch.log2(rank_alpha))

    # 计算 IDCG (假设正样本全在前面)
    idcg_relevant = torch.sort(labels, descending=True)[0][:k]
    idcg = torch.sum(idcg_relevant / torch.log2(rank_alpha))

    if idcg == 0: return 0.
    return (dcg / idcg).item()