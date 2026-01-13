# NRMS
参数的设置参考论文中的实现：epoch=10，batch_size=64，使用Adma优化器，Common Crawl 版本的 GloVe (300d, 840B tokens),1个正样本+4个负样本，如果不够则重复采样，超出随机抽样

## score on valid set
official——AUC:0.6275  nDCG@5:0.3217  nDCG@10:0.4139
original(epoch 10 lr=1e-3 )——AUC:0.6274  nDCG@5:0.3129  nDCG@10:0.3781
with ranger optimizer(epoch 10 lr=1e-3 )——AUC:0.6402  nDCG@5:0.3283  nDCG@10:0.3955
