
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
    """
    Set seed for reproducibility.
    
    Args:
        seed (int): Seed value to ensure consistent results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def GBF_distance_encode(matrix, min_value, max_value, step):
    """
    Gaussian-based function encoding for distance matrices.

    Args:
        matrix (numpy.array): Distance matrix to encode.
        min_value (float): Minimum value of encoding range.
        max_value (float): Maximum value of encoding range.
        step (int): Number of encoding steps.

    Returns:
        numpy.array: Encoded matrix with additional feature dimensions.
    """
    gamma = (max_value - min_value) / (step - 1)
    filters = np.linspace(min_value, max_value, step)
    matrix = matrix[:, :, np.newaxis]  # Add an extra dimension
    matrix = np.tile(matrix, (1, 1, step))  # Duplicate matrix along the new dimension
    matrix = np.exp(-((matrix - filters) ** 2) / gamma ** 2)  # Apply Gaussian filter
    return matrix


def mask_elements_over_threshold(matrix, threshold):
    """
    Mask elements in a matrix that are above a specified threshold by setting them to zero.

    Args:
        matrix (numpy.array): The input matrix to be masked.
        threshold (float): Threshold value for masking.

    Returns:
        numpy.array: Masked matrix with values over the threshold set to 0.
    """
    return np.where(matrix > threshold, 0.0, matrix)


class CharDataset(Dataset):
    def __init__(self, node_data, edge_data, graph_data, response, max_length, node_embd_len, graph_embd_len):
        """
        Dataset for handling node, edge, and graph data with energy as the response.

        Args:
            node_data (list): List of node embedding data.
            edge_data (list): List of edge adjacency matrices.
            graph_data (list): List of graph embeddings.
            response (list): List of target energies (responses).
            max_length (int): Maximum sequence length for padding.
            node_embd_len (int): Length of the node embedding vector.
            graph_embd_len (int): Length of the graph embedding vector.
        """
        self.data_size = len(node_data)
        print(f'Dataset contains {self.data_size} examples.')
        self.node_data = node_data
        self.edge_data = edge_data
        self.graph_data = graph_data
        self.response = response
        self.max_length = max_length
        self.node_embd_len = node_embd_len
        self.graph_embd_len = graph_embd_len

    def __len__(self):
        """Returns the size of the dataset."""
        return self.data_size

    def __getitem__(self, idx):
        """
        Returns the padded node, edge, graph, mask, and energy tensors for the specified index.

        Args:
            idx (int): Index of the data item to retrieve.

        Returns:
            tuple: Tuple of tensors:
                - node_padded: Padded node data.
                - edge_padded: Padded edge data.
                - graph: Graph data.
                - mask: Mask to indicate valid atoms.
                - energy: Energy value (target).
        """
        # Process node data
        node = np.array(self.node_data[idx])
        n_atoms = len(node)  # Number of atoms
        node_padded = torch.zeros(self.max_length, self.node_embd_len)
        node_padded[:n_atoms, :] = torch.tensor(node).float()  # Pad node embeddings

        # Process edge data
        edge = np.array(self.edge_data[idx])
        edge_rbf_bins = 12  # Number of radial basis function bins for edge encoding
        edge_extr = edge[:, :, np.newaxis]  # Add extra dimension for encoding
        edge_extr = np.repeat(edge_extr, edge_rbf_bins, axis=-1)
        edge = mask_elements_over_threshold(1.0 / edge, 8.0)  # Apply threshold mask
        edge_zero_extr = edge[:, :, np.newaxis]
        edge_zero_extr = np.repeat(edge_zero_extr, edge_rbf_bins, axis=-1)
        edge_embedding = GBF_distance_encode(edge, 0.0, 8.0, edge_rbf_bins)  # Encode distances
        edge_embedding[edge_extr == 1] = 1.0  # Handle special edge cases
        edge_embedding[edge_zero_extr == 0] = 0.0  # Handle zero cases
        edge_padded = torch.zeros(self.max_length, self.max_length, edge_rbf_bins)
        edge_padded[:n_atoms, :n_atoms, :] = torch.tensor(edge_embedding).float()  # Pad edges

        # Process graph embedding and mask
        graph = torch.tensor(np.array(self.graph_data[idx])).float()
        mask = torch.zeros(self.max_length).long()
        mask[:n_atoms] = 1  # Mask indicates which atoms are present
        energy = torch.tensor(np.array(self.response[idx])).float()

        return node_padded, edge_padded, graph, mask, energy.unsqueeze(0)  # Return energy with extra dimension
