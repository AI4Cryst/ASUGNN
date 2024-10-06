
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
from utils import set_seed, CharDataset
from models import *

# Configure logging to display timestamps and logging levels for better traceability
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def get_all_mpids(db):
    """
    Retrieve data from a database and extract necessary features for graph-based neural networks.

    Args:
        db (ase.db.Database): The database object to query.

    Returns:
        tuple: A tuple containing four lists:
            - node_data (list): Node embeddings from the database.
            - edge_data (list): Adjacency matrices (edges) from the database.
            - graph_data (list): Global graph-level features.
            - response (list): Target values (e.g., formation energy) for the model to predict.
    """
    entries = db.select()  # Query all entries in the database
    node_data, edge_data, graph_data, response = [], [], [], []

    # Progress bar for tracking loop execution
    pbar = tqdm(entries)
    for entry in pbar:
        data = entry.data
        formation_energy = entry.formation_energy
        node_data.append(data['node_embedding'])
        edge_data.append(data['adj_matrix'])
        graph_data.append(data['global_info'])
        response.append(formation_energy)
    
    return node_data, edge_data, graph_data, response


def load_model_and_predict(model_path, dataset, batch_size=64):
    """
    Load a pre-trained model, move it to the GPU, and perform predictions on a given dataset.

    Args:
        model_path (str): Path to the saved model.
        dataset (torch.utils.data.Dataset): Dataset to run predictions on.
        batch_size (int, optional): Batch size for DataLoader. Default is 64.

    Returns:
        tuple: Numpy arrays containing the model predictions and the corresponding ground truth.
    """
    prediction = []
    truth = []

    # Load the model from the specified path
    model = torch.load(model_path)
    model = model.to("cuda")  # Move the model to the GPU
    
    # Check for the available GPU device
    device = torch.cuda.current_device()
    logging.info(f'Device: {device}')

    # Use DataParallel for multi-GPU support, if available
    model = torch.nn.DataParallel(model).to(device)

    # Initialize DataLoader for predictions
    prediction_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2
    )
    
    # Progress bar for tracking prediction process
    pbar = tqdm(enumerate(prediction_dataloader), total=len(prediction_dataloader))
    for _, (nodes, edges, graph, mask, energy) in pbar:
        nodes, edges, graph, mask, energy = map(lambda x: x.cuda(), [nodes, edges, graph, mask, energy])

        # Disable gradient computation for inference
        with torch.no_grad():
            loss, pred = model(nodes, edges, graph, mask, energy)

        # Store predictions and ground truth
        prediction.append(pred.view(-1).detach())
        truth.append(energy.view(-1).detach())
    
    # Convert predictions and ground truth from tensors to numpy arrays
    pred = torch.cat(prediction).cpu().numpy()
    tru = torch.cat(truth).cpu().numpy()

    return pred, tru


if __name__ == "__main__":
    # Set GPU device environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

    # Set random seed for reproducibility
    set_seed(42)

    # Define dataset parameters
    max_length = 220
    node_embd_len = 106
    graph_embd_len = 140

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Predict formation energies using a trained model.")
    parser.add_argument(
        '--db_path', 
        type=str, 
        required=True, 
        help='Path to the SQLite database containing the crystal structures.'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        help='Path to the saved model file.'
    )
    
    args = parser.parse_args()  # Parse the arguments

    # Connect to the database and retrieve the necessary data
    db = connect(args.db_path)  # Use the database path from arguments
    node_data, edge_data, graph_data, response = get_all_mpids(db)

    # Create a dataset using the retrieved data
    dataset = CharDataset(node_data, edge_data, graph_data, response, max_length, node_embd_len, graph_embd_len)

    # Model prediction
    logging.info('Starting prediction...')
    pred, tru = load_model_and_predict(args.model_path, dataset)  # Use the model path from arguments

    # Print the results
    logging.info(f'Predictions: {pred}')
    logging.info(f'Truth: {tru}')
