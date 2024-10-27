"""
A simple training loop; a boilerplate that could apply to any arbitrary neural network.
This file does not specifically pertain to GPT models.
"""

import math
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

# Initialize logger
logger = logging.getLogger(__name__)

class TrainerConfig:
    """
    Configuration class for the Trainer that holds optimization parameters.
    """
    def __init__(self, **kwargs):
        # Default optimization parameters
        self.max_epochs = 10
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0
        self.weight_decay = 0.1  # Applied only on matmul weights
        self.lr_decay = False  # Enable learning rate decay
        self.warmup_tokens = 375e6  # Tokens for warmup phase
        self.final_tokens = 260e9  # Tokens at which we reach 10% of original LR
        self.ckpt_path = None  # Checkpoint path
        self.num_workers = 32  # Number of workers for DataLoader
        self.prefetch_factor = 2  # Prefetch factor for DataLoader
        
        # Update parameters from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    """
    Trainer class to handle the training and evaluation of the model.
    """
    def __init__(self, model, train_dataset, test_dataset, config, best_loss=None, device='gpu'):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # Set device based on availability of GPU
        self.device = 'cpu'
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print('Using GPU! Device={}'.format(self.device))

        self.best_loss = best_loss

    def save_checkpoint(self, epoch=''):
        """
        Save the model checkpoint to the specified path.
        
        Args:
            epoch (str): The current epoch number, used in the filename.
        """
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        path = self.config.ckpt_path.split('t.')
        logger.info("Saving model to %s", path[0] + epoch + '.' + path[1])
        torch.save(raw_model.state_dict(), path[0] + epoch + '.' + path[1])

    def train(self):
        """
        Main training loop that runs through epochs and handles training and evaluation.
        """
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        # Function to run a single epoch (training or testing)
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                prefetch_factor=config.prefetch_factor,
                                drop_last=False)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (nodes, edges, graph, mask, energy) in pbar:
                nodes = nodes.to(self.device)
                edges = edges.to(self.device)
                graph = graph.to(self.device)
                mask = mask.to(self.device)
                energy = energy.to(self.device)

                # Forward pass through the model
                with torch.set_grad_enabled(is_train):
                    loss, _ = model(nodes, edges, graph, mask, energy)
                    loss = loss.mean()
                    losses.append(loss.item())

                    if is_train:
                        # Backpropagation and parameter update
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                        # Learning rate decay
                        if config.lr_decay:
                            self.tokens += (nodes[:, 0, :] > 0).sum()  # Count processed tokens
                            if self.tokens < config.warmup_tokens:
                                # Linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # Cosine decay
                                progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.002, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # Update progress bar description
                        pbar.set_description(f"Epoch {epoch + 1} Iter {it}: Train loss {loss.item():.5f}. LR {lr:e}")
            torch.cuda.empty_cache()

            if not is_train:
                test_loss = float(np.mean(losses))
                print("Test loss: {}".format(test_loss))
                return test_loss

        # Initialize training parameters
        self.best_loss = float('inf') if self.best_loss is None else self.best_loss
        self.tokens = 0  # Counter for learning rate decay
        for epoch in range(config.max_epochs):
            run_epoch('train')
            self.save_checkpoint(epoch=str(epoch))
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # Early stopping based on test loss
            good_model = self.test_dataset is None or test_loss < self.best_loss
            if self.config.ckpt_path is not None and good_model:
                self.best_loss = test_loss
                self.save_checkpoint()
