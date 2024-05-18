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




# config
device='gpu'
numEpochs = 200 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
batchSize = 64 # batch size of training data
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
                      learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20,
                      final_tokens=2*len(train_dataset)*blockSize,
                      num_workers=num_workers, ckpt_path=ckptPath)
trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss, device=device)





print('Start training')
try:
    trainer.train()
except KeyboardInterrupt:
    print('KeyboardInterrupt')




# train_dataloader = DataLoader(
#     train_dataset,
#     shuffle=True,
#     batch_size=1,
#     num_workers=0
# )

# pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
# for i, (nodes, edges, graph, mask, energy) in pbar:
#     nodes = nodes.cuda()
#     edges = edges.cuda()
#     graph = graph.cuda()
#     mask = mask.cuda()
#     energy = energy.cuda()
#     loss, _,  = model(nodes, edges, graph, mask, energy)