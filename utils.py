# Author: Bin CAO <barniecao@outlook.com>

import json
import torch
import glob
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import random
from ase.db import connect
from ase.db.row import AtomsRow
from joblib import Parallel, delayed




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def GBF_distance_encode(matrix, min, max, step):
    """
    Encode the distance information in the input matrix using Gaussian-based functions.

    Args:
        matrix (numpy.array): The input matrix containing distance information.
        min (float): The minimum value for encoding.
        max (float): The maximum value for encoding.
        step (int): The number of steps for encoding.

    Returns:
        numpy.array: The encoded matrix with distance information.
    """
    gamma = (max - min) / (step - 1)
    filters = np.linspace(min, max, step)
    matrix = matrix[:, :, np.newaxis]
    matrix = np.tile(matrix, (1, 1, step))
    matrix = np.exp(-((matrix - filters) ** 2) / gamma**2)
    return matrix




def mask_elements_over_threshold(matrix, threshold):
    """
    A function that masks elements in a matrix that are over a certain threshold.

    Args:
        matrix (numpy.array): The input matrix to be masked.
        threshold (float): The threshold value for masking.

    Returns:
        numpy.array: The masked matrix with elements over the threshold set to 0.0.
    """
    masked_matrix = np.where(matrix > threshold, 0.0, matrix)
    return masked_matrix


class CharDataset(Dataset):
    def __init__(self, node_data, edge_data, graph_data, response, max_length, node_embd_len, graph_embd_len):
        """
        Initializes a new instance of the CharDataset class.

        Args:
            node_data (list): A list of node data.
            edge_data (list): A list of edge data.
            graph_data (list): A list of graph data.
            response (list): A list of responses.
            max_length (int): The maximum length of the data.
            node_embd_len (int): The length of the node embedding.
            graph_embd_len (int): The length of the graph embedding.

        Returns:
            None
        """

        self.data_size = len(node_data)
        print('data has %d examples' % (self.data_size))
        self.node_data = node_data
        self.edge_data = edge_data
        self.graph_data = graph_data
        self.response = response
        self.max_length = max_length
        self.node_embd_len = node_embd_len
        self.graph_embd_len = graph_embd_len

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        """
        Returns a tuple containing padded node, edge, graph, mask, and energy tensors for a given index.
        
        :param idx: An integer representing the index of the data to retrieve.
        :type idx: int
        :return: A tuple containing the following tensors:
                    - node_padded: A tensor of shape (max_length, node_embd_len) representing padded node data.
                    - edge_padded: A tensor of shape (max_length, max_length, edge_rbf_bins) representing padded edge data.
                    - graph: A tensor of shape (graph_embd_len) representing graph data.
                    - mask: A tensor of shape (max_length) representing a mask indicating the presence of atoms.
                    - energy: A tensor of shape (1) representing the energy of the system.
        :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        """
        # go to node_embedding
        node = np.array(self.node_data[idx])
        n_atoms = len(node)
        node_padded = torch.zeros(self.max_length, self.node_embd_len)
        node_padded[:n_atoms, :] = torch.tensor(node).float()

        # go to edge_embedding
        edge = np.array(self.edge_data[idx])
        edge_rbf_bins = 12
        edge_extr = edge[:, :, np.newaxis]
        edge_extr = np.repeat(edge_extr, edge_rbf_bins, axis=-1)
        edge = mask_elements_over_threshold(1.0 / edge, 8.0)
        edge_zero_extr = edge[:, :, np.newaxis]
        edge_zero_extr = np.repeat(edge_zero_extr, edge_rbf_bins, axis=-1)
        edge_embedding = GBF_distance_encode(edge, 0.0, 8.0, edge_rbf_bins)
        edge_embedding[edge_extr == 1] = 1.0
        edge_embedding[edge_zero_extr == 0] = 0.0
        edge_padded = torch.zeros(self.max_length, self.max_length, edge_rbf_bins)
        edge_padded[:n_atoms, :n_atoms, :] = torch.tensor(edge_embedding).float()

        # go to graph_embedding
        graph = torch.tensor(np.array(self.graph_data[idx])).float()
        mask = torch.zeros(self.max_length).long()
        mask[:n_atoms] = 1
        energy = torch.tensor(np.array(self.response[idx])).float()
        return node_padded, edge_padded, graph, mask, energy.unsqueeze(0)

