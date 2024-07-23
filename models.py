import math
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)




class NetConfig:
    """ base PointNet config """

    def __init__(self, embeddingSize, numberofPoints,
                 **kwargs):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # number of points


        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        """
        Initializes the CausalSelfAttention module.

        Args:
            config (object): The configuration object containing the following attributes:
                - n_embd (int): The embedding size.
                - n_head (int): The number of attention heads.
                - attn_pdrop (float): The dropout probability for the attention mechanism.
                - resid_pdrop (float): The dropout probability for the residual connections.

        Raises:
            AssertionError: If n_embd is not divisible by n_head.

        Initializes the following attributes:
            - key (nn.Linear): The linear layer for projecting the input to the key space.
            - query (nn.Linear): The linear layer for projecting the input to the query space.
            - value (nn.Linear): The linear layer for projecting the input to the value space.
            - attn_drop (nn.Dropout): The dropout layer for the attention mechanism.
            - resid_drop (nn.Dropout): The dropout layer for the residual connections.
            - proj (nn.Linear): The linear layer for projecting the output back to the input space.
            - adj_proj1 (nn.Linear): The linear layer for projecting the adjacency matrix to the attention weights.
            - adj_act (nn.GELU): The activation function for the attention weights.
            - adj_proj2 (nn.Linear): The linear layer for projecting the attention weights to the final attention weights.
            - adj_weight (nn.Parameter): The learnable parameter for the attention weights.
            - n_head (int): The number of attention heads.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.adj_proj1 = nn.Linear(12, 2 * config.n_head, bias=False)
        self.adj_act = nn.GELU()
        self.adj_proj2 = nn.Linear(2 * config.n_head, config.n_head, bias=False)
        self.adj_weight = nn.Parameter(torch.tensor(0.00001))
        self.n_head = config.n_head

    def forward(self, x, x_padding_judge, adj, is_formula=None):
        """
        Perform the forward pass of the model.

        Parameters:
            x (torch.Tensor): The input tensor.
            x_padding_judge (torch.Tensor): The padding mask tensor.
            adj (torch.Tensor): The adjacency matrix tensor.
        """
        B, T, C = x.size()
        x_padding_judge = 1.0 - x_padding_judge
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=2)
        x_padding_judge = x_padding_judge @ x_padding_judge.transpose(-2, -1)
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=1)
        x_padding_judge = torch.tile(x_padding_judge, [1, self.n_head, 1, 1])
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_fix = self.adj_act(self.adj_proj1(adj))
        att_fix = self.adj_proj2(att_fix)
        att_fix = torch.einsum('ijkl->iljk', att_fix)
        att_full = torch.ones_like(att)
        att_full = self.attn_drop(att_full)
        x_padding_judge = x_padding_judge * att_full
        att = att.masked_fill(x_padding_judge == 0, -1e9)
        att = F.softmax(att, dim=-1)
        att = att + self.adj_weight * att_fix
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y


class CausalCrossAttention(nn.Module):
    def __init__(self, config):
        """
        Initializes the CausalCrossAttention module.
        Args:
            config (object): The configuration object containing the following attributes:
                - n_embd (int): The embedding size.
                - n_head (int): The number of attention heads.
                - attn_pdrop (float): The dropout probability for the attention mechanism.
                - resid_pdrop (float): The dropout probability for the residual connections.

        Raises:
            AssertionError: If n_embd is not divisible by n_head.

        Initializes the following attributes:
            - key (nn.Linear): The linear layer for projecting the input to the key space.
            - query (nn.Linear): The linear layer for projecting the input to the query space.
            - value (nn.Linear): The linear layer for projecting the input to the value space.
            - attn_drop (nn.Dropout): The dropout layer for the attention mechanism.
            - resid_drop (nn.Dropout): The dropout layer for the residual connections.
            - proj (nn.Linear): The linear layer for projecting the output back to the input space.
            - adj_proj1 (nn.Linear): The linear layer for projecting the adjacency matrix to the attention weights.
            - adj_act (nn.GELU): The activation function for the attention weights.
            - adj_proj2 (nn.Linear): The linear layer for projecting the attention weights to the final attention weights.
            - adj_weight (nn.Parameter): The learnable parameter for the attention weights.
            - n_head (int): The number of attention heads.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.adj_proj1 = nn.Linear(1, 2 * config.n_head, bias=False)
        self.adj_act = nn.GELU()
        self.adj_proj2 = nn.Linear(2 * config.n_head, config.n_head, bias=False)
        self.adj_weight = nn.Parameter(torch.tensor(50.0))
        self.n_head = config.n_head

    def forward(self, x, x_padding_judge, adj, graph, is_formula=None):
        B, T, C = x.size()
        k = self.key(graph).view(B, 1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(graph).view(B, 1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y


class ASU_codec_block(nn.Module):
    def __init__(self, config):
        """
        Initializes the ASU_codec_block module with the given configuration.
        
        Args:
            config (object): The configuration object containing the following attributes:
                - n_embd (int): The embedding size.
                - resid_pdrop (float): The dropout probability for the residual connections.
        
        Initializes the following attributes:
            - ln1 (nn.LayerNorm): Layer normalization for the first layer.
            - ln2 (nn.LayerNorm): Layer normalization for the second layer.
            - attn1 (CausalSelfAttention): The causal self-attention module.
            - mlp (nn.Sequential): The multi-layer perceptron module.
            - ln3 (nn.LayerNorm): Layer normalization for the third layer.
            - ln4 (nn.LayerNorm): Layer normalization for the fourth layer.
            - attn2 (CausalCrossAttention): The causal cross-attention module.
            - mlp2 (nn.Sequential): The second multi-layer perceptron module.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn1 = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        self.attn2 = CausalCrossAttention(config)
        self.mlp2 = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, x_padding_judge, adj, graph, is_formula):
        """
        Perform the forward pass of the model.
        
        Parameters:
            x (torch.Tensor): The input tensor.
            x_padding_judge (torch.Tensor): The padding mask tensor.
            adj (torch.Tensor): The adjacency matrix tensor.
            graph: The graph tensor.
            is_formula (bool): A boolean flag indicating if it is a formula.
        
        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        x = x + self.attn1(self.ln1(x), x_padding_judge, adj, is_formula)
        x = x + self.mlp(self.ln2(x))
        x = x + self.attn2(self.ln3(x), x_padding_judge, adj, graph, is_formula)
        x = x + self.mlp2(self.ln4(x))
        return x

class ASU_Codec(nn.Module):
    def __init__(self, config):
        """
        Initializes the ASU_Codec class with a list of ASU_codec_block instances based on the specified configuration.
        
        Parameters:
            config: Configuration object containing parameters for initializing the ASU_Codec.
        """
        super().__init__()
        self.blocks = nn.ModuleList([ASU_codec_block(config) for _ in range(config.n_layer)])
    def forward(self, x, x_padding_judge, adj, graph):
        for block in self.blocks:
            x = block(x, x_padding_judge, adj, graph, is_formula=True)
        return x



class ASUGNN(nn.Module):
    def __init__(self, config, pointNetConfig=None):
        """
        Initializes the ASUGNN model.

        Args:
            config (Config): The configuration object containing the model parameters.
        Initializes the following attributes:
            - config (Config): The configuration object.
            - pointNet (None): The PointNet model.
            - embeddingSize (int): The size of the embedding.
            - max_length (int): The maximum length of the input sequence.
            - dropkey (nn.Dropout): The dropout layer for the attention scores.
            - att_score (nn.Linear): The linear layer for computing attention scores.
            - element_embd (nn.Linear): The linear layer for element embedding.
            - cgcnn_fc1 (nn.Linear): The first linear layer for CGCNN.
            - cgcnn_act1 (nn.Sigmoid): The activation function for CGCNN.
            - Trans_element_encoder (ASU_formula_encoder): The  encoder for element encoding.
            - ln_before_aggregate (nn.LayerNorm): The layer normalization before aggregation.
            - ln_after_aggregate (nn.LayerNorm): The layer normalization after aggregation.
        """
        
        super().__init__()

        self.config = config
        self.pointNetConfig = pointNetConfig
        self.pointNet = None
        embeddingSize = config.n_embd
        self.max_length = config.max_length
        self.ln_gru = nn.LayerNorm(embeddingSize)
        self.dropkey = nn.Dropout(config.attn_pdrop)
        self.gru = nn.GRU(bidirectional=True, hidden_size=embeddingSize, batch_first=True, input_size=embeddingSize)
        self.att_score = nn.Linear(2 * embeddingSize, 1, bias=False)
        self.element_embd = nn.Linear(config.node_embd_len, 64)
        self.cgcnn_fc1 = nn.Linear(64*2+12, 64*2)
        self.cgcnn_act1 = nn.Sigmoid()
        self.cgcnn_act2 = nn.Softplus()
        self.cgcnn_bn11 = nn.BatchNorm1d(2*64)
        self.cgcnn_bn12 = nn.BatchNorm1d(64)
        self.cgcnn_fc2 = nn.Linear(64*2+12, 64*2)
        self.cgcnn_bn21 = nn.BatchNorm1d(2*64)
        self.cgcnn_bn22 = nn.BatchNorm1d(64)
        self.graph_convert_trans = nn.Linear(64, embeddingSize)
        self.graph_embd = nn.Sequential(
            nn.Linear(config.graph_embd_len, 2 * config.n_embd),
            nn.GELU(),
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.Trans_element_encoder = ASU_Codec(config)
        self.ln_before_aggregate = nn.LayerNorm(config.n_embd)
        self.ln_after_aggregate = nn.LayerNorm(2 * config.n_embd)
        self.out1 = nn.Linear(2 * config.n_embd, 4 * config.n_embd)
        self.out_act = nn.ReLU()
        self.out2 = nn.Linear(4 * config.n_embd, 1)
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        """
        Returns the block size of the model.
        """
        return self.block_size

    def _init_weights(self, module):
        """
        Initialize the weights of the given module.
            None
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        Separates the model's parameters into two buckets: those that will and won't experience regularizing weight decay.
        It then returns a PyTorch optimizer object with the specified parameters and configuration.
        
        Args:
            train_config (object): An object containing the training configuration.
                - learning_rate (float): The learning rate for the optimizer.
                - weight_decay (float): The weight decay for the optimizer.
                - betas (tuple): The betas for the optimizer.
        
        Returns:
            torch.optim.AdamW: The AdamW optimizer object.
        
        Raises:
            AssertionError: If there are parameters that are in both the decay and no_decay sets.
            AssertionError: If there are parameters that are not in either the decay or no_decay sets.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.GRU)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias') or 'gru' in mn or 'logit_scale' in pn:
                    no_decay.add(fpn)
                elif 'adj_weight' in pn:  # Process all 'adj_weight' separately
                    decay.add(fpn)  # Assume we decide to apply weight decay to all 'adj_weight'
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, nodes, edges, graph, mask, energy):
        nodes = self.element_embd(nodes)
        b, t, e = nodes.size()
        nodes_i = nodes.unsqueeze(2).expand(b, t, t, e)
        nodes_j = nodes.unsqueeze(1).expand(b, t, t, e)
        nodes_res = nodes
        nodes = torch.cat([nodes_i, nodes_j, edges], dim=-1)
        nodes = self.cgcnn_fc1(nodes)
        nodes = self.cgcnn_bn11(nodes.view(-1, 2 * e)).view(b, t, t, 2 * e)
        nodes_filter, nodes_core = nodes.chunk(2, dim=-1)
        nodes_filter = self.cgcnn_act1(nodes_filter)
        nodes_core = self.cgcnn_act2(nodes_core)
        nodes_filter = nodes_filter.masked_fill(mask.unsqueeze(-1).unsqueeze(1).expand(b, t, t, e) == 0, 0.0)
        nodes_filter = nodes_filter.masked_fill(mask.unsqueeze(-1).unsqueeze(2).expand(b, t, t, e) == 0, 0.0)
        nodes_core = nodes_core.masked_fill(mask.unsqueeze(-1).unsqueeze(1).expand(b, t, t, e) == 0, 0.0)
        nodes_core = nodes_core.masked_fill(mask.unsqueeze(-1).unsqueeze(2).expand(b, t, t, e) == 0, 0.0)
        nodes = torch.sum(nodes_filter * nodes_core, dim=2)
        nodes = self.cgcnn_bn12(nodes.view(-1, e)).view(b, t, e)
        nodes = nodes_res + self.cgcnn_act2(nodes)
        nodes_i = nodes.unsqueeze(2).expand(b, t, t, e)
        nodes_j = nodes.unsqueeze(1).expand(b, t, t, e)
        nodes_res = nodes
        nodes = torch.cat([nodes_i, nodes_j, edges], dim=-1)
        nodes = self.cgcnn_fc2(nodes)
        nodes = self.cgcnn_bn21(nodes.view(-1, 2 * e)).view(b, t, t, 2 * e)
        nodes_filter, nodes_core = nodes.chunk(2, dim=-1)
        nodes_filter = self.cgcnn_act1(nodes_filter)
        nodes_core = self.cgcnn_act2(nodes_core)
        nodes_filter = nodes_filter.masked_fill(mask.unsqueeze(-1).unsqueeze(1).expand(b, t, t, e) == 0, 0.0)
        nodes_filter = nodes_filter.masked_fill(mask.unsqueeze(-1).unsqueeze(2).expand(b, t, t, e) == 0, 0.0)
        nodes_core = nodes_core.masked_fill(mask.unsqueeze(-1).unsqueeze(1).expand(b, t, t, e) == 0, 0.0)
        nodes_core = nodes_core.masked_fill(mask.unsqueeze(-1).unsqueeze(2).expand(b, t, t, e) == 0, 0.0)
        nodes = torch.sum(nodes_filter * nodes_core, dim=2)
        nodes = self.cgcnn_bn22(nodes.view(-1, e)).view(b, t, e)
        nodes = nodes_res + self.cgcnn_act2(nodes)
        nodes = nodes.masked_fill(mask.unsqueeze(-1).expand(b, t ,e) == 0, 0.0)
        x = self.graph_convert_trans(nodes)  # each index maps to a (learnable) vector -> b x length x embedding
        graph = self.graph_embd(graph)
        mask = 1 - mask
        energy = energy.unsqueeze(0)
        x = self.Trans_element_encoder(x, mask, edges, graph)
        x = self.ln_before_aggregate(x)

        # aggregate the ASUGNN output
        self.gru.flatten_parameters()
        x, _ = self.gru(self.ln_gru(x))
        att_score = self.att_score(x)
        mask = 1.0 - mask.float()
        att_score = torch.where(mask==0.0, torch.ones_like(mask) * -1e10, att_score.squeeze(-1))
        att_weights = torch.softmax(att_score, dim=1)
        x = torch.sum(att_weights.unsqueeze(-1) * x, dim=1)
        x = self.out_act(self.out1(x))
        pred = self.out2(x).squeeze()
        loss = nn.functional.mse_loss(pred.view(-1), energy.view(-1))
        return loss, pred
