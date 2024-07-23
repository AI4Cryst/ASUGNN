import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

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
from ase.db import connect
from utils import set_seed, CharDataset
from models import *


def get_all_mpids(db):
    """
    Args:
        db (ase.db.Database): The database object to query.

    Returns:
        tuple: A tuple containing four lists:
            - node_data (list): A list of node embeddings.
            - edge_data (list): A list of adjacency matrices.
            - graph_data (list): A list of global information.
            - response (list): A list of formation energies.
    """
    entries = db.select()
    node_data, edge_data, graph_data, response = [], [], [], []

    pbar = tqdm(entries)
    for entry in pbar:
        all_we_need = [entry.data, entry.formation_energy]
        node_data.append(all_we_need[0]['node_embedding'])
        edge_data.append(all_we_need[0]['adj_matrix'])
        graph_data.append(all_we_need[0]['global_info'])
        response.append(all_we_need[1])
    return node_data, edge_data, graph_data, response


def load_model_and_predict(model_path, dataset, batch_size=64):
    """
    Load a model, move it to GPU, and perform predictions on a dataset.

    Parameters:
    model_path (str): Path to the saved model.
    dataset (torch.utils.data.Dataset): Dataset for prediction.
    batch_size (int): Batch size for DataLoader. Default is 64.

    Returns:
    tuple: Predictions and ground truth as numpy arrays.
    """
    prediction = []
    truth = []
    model = torch.load(model_path)
    model = model.to("cuda")
    device = torch.cuda.current_device()
    print('device:{}'.format(device))
    model = torch.nn.DataParallel(model).to(device)
    prediction_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2)
    pbar = tqdm(enumerate(prediction_dataloader), total=len(prediction_dataloader))
    for _, (nodes, edges, graph, mask, energy) in pbar:
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
    return pred, tru



os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
set_seed(42)
max_length = 220# 100
node_embd_len = 106
graph_embd_len = 140
db = connect('/home/cb/cb_crystal/test_ASUnet/filter_self_struc_cif.db')  # Replace this with your database path
node_data, edge_data, graph_data, response = get_all_mpids()
dataset = CharDataset(node_data, edge_data, graph_data, response, max_length, node_embd_len, graph_embd_len)

if __name__ == "__main__":
    model_path = 'model.pt'
    dataset = ...  # Replace this with your dataset instance
    pred, tru = load_model_and_predict(model_path, dataset)