import glob
import os
import numpy as np
import copy as cpy
import torch
import random
import pandas as pd
import multiprocessing as mp
from tqdm import  tqdm
from torch.utils.data import DataLoader, random_split
# from torch.utils.data import DataLoader
# from ase.db import connect
from trainer import Trainer, TrainerConfig
from utils import set_seed, CharDataset
from models import GPT, GPTConfig, PointNetConfig


set_seed(42)


node_data = np.load('./embedding/node_emd.npy', allow_pickle=True).tolist()
edge_data = np.load('./embedding/adj_matrix.npy', allow_pickle=True).tolist()
graph_data = np.load('./embedding/global_emd.npy', allow_pickle=True).tolist()
response = np.load('./embedding/response.npy', allow_pickle=True).tolist()


idxes = [id for id in range(0, len(node_data)) if len(node_data[id]) < 500]
node_data = [node_data[i] for i in idxes]
edge_data = [edge_data[i] for i in idxes]
graph_data = [graph_data[i] for i in idxes]
response = [response[i] for i in idxes]

# set the random seed

# config
device='gpu'
numEpochs = 200 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
batchSize = 64  # batch size of training data
target = 'Skeleton' #'Skeleton' #'EQ'
decimals = 4 # decimals
n_layers = 2
n_heads = 8
numPoints = 9
blockSize = 95
num_workers = 42
dataInfo = 'Layers_{}Heads_{}EmbeddingSize{}'.format(n_layers, n_heads, embeddingSize)
addr = './SavedModels/' # where to save model
bestLoss = None # if there is any model to load as pre-trained one
fName = '{}_permittivity_clip_MINIMIZE_gbf_test.txt'.format(dataInfo)
ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])



max_length = 500
node_embd_len = 106
graph_embd_len = 700

dataset = CharDataset(node_data, edge_data, graph_data, response, max_length, node_embd_len, graph_embd_len)


total_size = len(dataset)
train_size = int(total_size * 0.7)
test_size = int(total_size * 0.15)
# 确保训练集、测试集和验证集的总和等于数据集的总大小
validation_size = total_size - train_size - test_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])


pconf = PointNetConfig(embeddingSize=embeddingSize,
                       numberofPoints=numPoints)
mconf = GPTConfig(max_length, node_embd_len, graph_embd_len,
                  n_layer=n_layers, n_head=n_heads, n_embd=embeddingSize)
model = GPT(mconf, pconf)

# 把model保存下来
torch.save(model, './unpara_model.pth')
# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=numEpochs, batch_size=batchSize,
                      learning_rate=1e-3,
                      lr_decay=True, warmup_tokens=512*20,
                      final_tokens=2*len(train_dataset)*blockSize,
                      num_workers=num_workers, ckpt_path=ckptPath)
trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss, device=device)





print('Start training')
# try:
#     trainer.train()
# except KeyboardInterrupt:
#     print('KeyboardInterrupt')




# load the best model
print('The following model {} has been loaded!'.format(ckptPath))
model.load_state_dict(torch.load(ckptPath))
model = model.to(trainer.device)

device = torch.cuda.current_device()
print('device:{}',format(device))
model = torch.nn.DataParallel(model).to(device)


train_dataloader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=64*4,
    num_workers=32,
    drop_last=True
)

prediction = []
truth = []
pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
for i, (nodes, edges, graph, mask, energy) in pbar:
    nodes = nodes.cuda()
    edges = edges.cuda()
    graph = graph.cuda()
    mask = mask.cuda()
    energy = energy.cuda()
    with torch.no_grad():
        loss, pred = model(nodes, edges, graph, mask, energy)
    prediction.append(pred.view(-1).detach())
    truth.append(energy.view(-1).detach())
    


pred = torch.cat(prediction).cpu().numpy()
tru = torch.cat(truth).cpu().numpy()

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# 示例数据：真实值和预测值
y_true = tru
y_pred = pred

# 计算R-squared值
r2 = r2_score(y_true, y_pred)
mse = mean_absolute_error(y_true, y_pred)

errors = np.abs(y_true - y_pred)
mae_threshold = 0.02

acc = len(np.where(errors < mae_threshold)[0]) / len(tru)
print('acc:{}'.format(acc))

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, color='blue')
plt.title('Prediction vs. True Value')
plt.xlabel('True Values')
plt.ylabel('Predictions')

# 绘制y=x参考线，以便直观看出预测值与真实值的偏差
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)

# 添加R-squared值到图表上
plt.text(y_true.min(), y_pred.max(), f'R^2 = {r2:.2f}\nMAE = {mse:.2f}', fontsize=12)

plt.legend()

plt.show()

plt.savefig('prediction_vs_true_value.svg', format='svg', dpi=600)

