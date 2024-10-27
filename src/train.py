import argparse
import logging
import glob
import os
import numpy as np
import copy as cpy
import torch
import random
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from ase.db import connect
from trainer import Trainer, TrainerConfig
from utils import set_seed, CharDataset
from models import GPT, GPTConfig, PointNetConfig

# Setting up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--db_path', type=str, required=True, help='Path to the parsed database.')
parser.add_argument('--model_saving_path', type=str, required=True, help='Path to save the model.')
parser.add_argument('--batchsize', type=int, default=32, help='Batch size for training.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of training epochs.')
parser.add_argument('--device', type=str, default='cpu', help='Device for training (cpu or gpu).')

args = parser.parse_args()

# Use args to access command-line arguments
db_path = args.db_path
model_saving_path = args.model_saving_path
batch_size = args.batchsize
learning_rate = args.lr
num_epochs = args.num_epoch
device = args.device

# Set random seed for reproducibility
set_seed(42)
db = connect(db_path)

def get_all_mpids():
    """
    Retrieve all entries from the database and extract relevant data.
    
    Returns:
        node_data (list): List of node embeddings.
        edge_data (list): List of adjacency matrices.
        graph_data (list): List of global graph information.
        response (list): List of formation energies.
    """
    # Select all entries from the database
    entries = db.select()
    
    # Initialize lists to store data
    node_data, edge_data, graph_data, response = [], [], [], []
    
    # Progress bar for visual feedback
    pbar = tqdm(entries)
    
    # Extract data from each entry
    for entry in pbar:
        all_we_need = [entry.data, entry.formation_energy]
        node_data.append(all_we_need[0]['node_embedding'])
        edge_data.append(all_we_need[0]['adj_matrix'])
        graph_data.append(all_we_need[0]['global_info'])
        response.append(all_we_need[1])

    return node_data, edge_data, graph_data, response

# Call the function to get all mpid labels
node_data, edge_data, graph_data, response = get_all_mpids()

# Configuration for the model
num_epochs = num_epochs  # Number of epochs to train the model
embedding_size = batch_size  # Hidden dimension for embeddings
batch_size = 512  # Batch size for training data
target = 'Skeleton'  # Target variable
decimals = 4  # Precision for numerical output
n_layers = 2  # Number of layers in the model
n_heads = 2  # Number of attention heads
num_points = 9  # Number of points for PointNet
block_size = 95  # Size of the input blocks
num_workers = 4  # Number of worker threads for data loading

# Data information for saving purposes
data_info = 'Layers_{}Heads_{}EmbeddingSize{}'.format(n_layers, n_heads, embedding_size)
addr = model_saving_path  # Directory to save the model
best_loss = None  # Variable to store the best loss for early stopping

# Model file naming and checkpoint path
f_name = '{}_permittivity_clip_MINIMIZE_gbf_test.txt'.format(data_info)
ckpt_path = '{}/{}.pt'.format(addr, f_name.split('.txt')[0])

# Lengths for embedding configurations
max_length = 220
node_embd_len = 106
graph_embd_len = 140

# Create dataset object
dataset = CharDataset(node_data, edge_data, graph_data, response, max_length, node_embd_len, graph_embd_len)

# Split dataset into training, validation, and test sets
total_size = len(dataset)
train_size = int(total_size * 0.7)
test_size = int(total_size * 0.15)
validation_size = total_size - train_size - test_size

_, val_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
train_dataset = dataset

# Initialize model configuration
pconf = PointNetConfig(embeddingSize=embedding_size, numberofPoints=num_points)
mconf = GPTConfig(max_length, node_embd_len, graph_embd_len, n_layer=n_layers, n_head=n_heads, n_embd=embedding_size)
model = GPT(mconf, pconf)

# Save the model configuration
torch.save(model, './unpara_model.pth')

# Initialize trainer instance and configure training parameters
tconf = TrainerConfig(
    world_size=8,
    max_epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    lr_decay=True,
    warmup_tokens=512 * 20,
    final_tokens=2 * len(train_dataset) * block_size,
    num_workers=num_workers,
    ckpt_path=ckpt_path
)

trainer = Trainer(model, train_dataset, val_dataset, tconf, best_loss, device=device)

if __name__ == '__main__':
    print('Start training')
    try:
        trainer.train()
    except KeyboardInterrupt:
        print('Training interrupted by user.')
