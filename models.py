"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, max_length, node_embd_len, graph_embd_len, **kwargs):
        self.max_length = max_length
        self.node_embd_len = node_embd_len
        self.graph_embd_len = graph_embd_len
        for k, v in kwargs.items():
            setattr(self, k, v)



class PointNetConfig:
    """ base PointNet config """

    def __init__(self, embeddingSize, numberofPoints,
                 **kwargs):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # number of points


        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention_masked_for_formula(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
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
        self.adj_weight = nn.Parameter(torch.tensor(50.0))
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, x_padding_judge, adj, is_formula=None):
        B, T, C = x.size()

        x_padding_judge = 1.0 - x_padding_judge
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=2)
        x_padding_judge = x_padding_judge @ x_padding_judge.transpose(-2, -1)
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=1)
        x_padding_judge = torch.tile(x_padding_judge, [1, self.n_head, 1, 1])

        ############# 额外cls ##############################
        # if is_formula:
        #     x_padding_judge[:, :, 7:, 0] = 0.0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        ######## use adj to fix att  ####################
        # adj = adj.unsqueeze(-1)
        att_fix = self.adj_act(self.adj_proj1(adj))
        att_fix = self.adj_proj2(att_fix)
        att_fix = torch.einsum('ijkl->iljk', att_fix)

        ##########Dropkey#####################
        att_full = torch.ones_like(att)
        att_full = self.attn_drop(att_full)
        x_padding_judge = x_padding_judge * att_full
        ############################################
        att = att.masked_fill(x_padding_judge == 0, -1e9)
        att = F.softmax(att, dim=-1)

        att = att + self.adj_weight * att_fix
        # att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y




class CausalCrossAttention_masked_for_formula(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
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
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, x_padding_judge, adj, graph, is_formula=None):
        B, T, C = x.size()

        # x_padding_judge = 1.0 - x_padding_judge
        # x_padding_judge = torch.unsqueeze(x_padding_judge, dim=2)
        # x_padding_judge = x_padding_judge @ x_padding_judge.transpose(-2, -1)
        # x_padding_judge = torch.unsqueeze(x_padding_judge, dim=1)
        # x_padding_judge = torch.tile(x_padding_judge, [1, self.n_head, 1, 1])

        ############# 额外cls ##############################
        # if is_formula:
        #     x_padding_judge[:, :, 7:, 0] = 0.0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(graph).view(B, 1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(graph).view(B, 1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        ##########Dropkey#####################
        # att_full = torch.ones_like(att)
        # att_full = self.attn_drop(att_full)
        # x_padding_judge = x_padding_judge * att_full
        ############################################
        # att = att.masked_fill(x_padding_judge == 0, -1e9)
        att = F.softmax(att, dim=-1)

        # att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y



class Transformer_encoder_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn1 = CausalSelfAttention_masked_for_formula(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        self.attn2 = CausalCrossAttention_masked_for_formula(config)
        self.mlp2 = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )



    def forward(self, x, x_padding_judge, adj, graph, is_formula):
        x = x + self.attn1(self.ln1(x), x_padding_judge, adj, is_formula)
        x = x + self.mlp(self.ln2(x))
        x = x + self.attn2(self.ln3(x), x_padding_judge, adj, graph, is_formula)
        x = x + self.mlp2(self.ln4(x))
        return x




class Transformer_point_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Transformer_encoder_block(config) for _ in range(config.n_layer)])

    def forward(self, x, x_padding_judge):
        for block in self.blocks:
            x = block(x, x_padding_judge, is_formula=False)
        return x







class Transformer_formula_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Transformer_encoder_block(config) for _ in range(config.n_layer)])

    def forward(self, x, x_padding_judge, adj, graph):
        for block in self.blocks:
            x = block(x, x_padding_judge, adj, graph, is_formula=True)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model)  # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1)  # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *  # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)  # size = [batch, L, d_model]
        return x  # size = [batch, L, d_model]







# pointNet based on Convolution, T-NET naming is not accurate
class tNet(nn.Module):
    """
    The PointNet structure in the orginal PointNet paper:
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et. al. 2017
    """
    def __init__(self, config):
        super(tNet, self).__init__()

        self.activation_func = F.relu
        self.num_units = config.embeddingSize

        self.conv1 = nn.Conv1d(config.numberofVars+config.numberofYs, self.num_units, 1)
        self.conv2 = nn.Conv1d(self.num_units, 2 * self.num_units, 1)
        self.conv3 = nn.Conv1d(2 * self.num_units, 4 * self.num_units, 1)
        self.fc1 = nn.Linear(4 * self.num_units, 2 * self.num_units)
        self.fc2 = nn.Linear(2 * self.num_units, self.num_units)

        #self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(config.numberofVars+config.numberofYs)
        #self.input_layer_norm = nn.LayerNorm(config.numberofPoints)

        self.bn1 = nn.BatchNorm1d(self.num_units)
        self.bn2 = nn.BatchNorm1d(2 * self.num_units)
        self.bn3 = nn.BatchNorm1d(4 * self.num_units)
        self.bn4 = nn.BatchNorm1d(2 * self.num_units)
        self.bn5 = nn.BatchNorm1d(self.num_units)

    def forward(self, x):
        """
        :param x: [batch, #features, #points]
        :return:
            logit: [batch, embedding_size]
        """
        x = self.input_batch_norm(x)
        x = self.activation_func(self.bn1(self.conv1(x)))
        x = self.activation_func(self.bn2(self.conv2(x)))
        x = self.activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4 * self.num_units

        x = self.activation_func(self.bn4(self.fc1(x)))
        x = self.activation_func(self.bn5(self.fc2(x)))
        #x = self.fc2(x)

        return x




class points_emb(nn.Module):
    def __init__(self, config, in_channels=3):
        super(points_emb, self).__init__()
        emb_every = config.n_embd // in_channels
        emb_remainder = config.n_embd % in_channels
        self.fc1 = nn.Linear(1, emb_every)
        self.fc2 = nn.Linear(1, emb_every)
        self.fc3 = nn.Linear(1, emb_every + emb_remainder)



    def forward(self, xyz):
        xyz = xyz.transpose(dim0=1, dim1=2)
        out1 = self.fc1(xyz[:,:,0:1])
        out2 = self.fc2(xyz[:,:,1:2])
        out3 = self.fc3(xyz[:,:,2:3])
        points_emb = torch.cat([out1, out2, out3], dim=2)
        return points_emb



class seq_emb(nn.Module):
    def __init__(self, config):
        super(seq_emb, self).__init__()
        self.fc1 = nn.Linear(1, config.n_embd)

    def forward(self, seq):
        seq = seq.unsqueeze(dim=2)
        seq_emb = self.fc1(seq)
        return seq_emb



class project_mlp(nn.Module):
    def __init__(self, config):
        super(project_mlp, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.fc2 = nn.Linear(2 * config.n_embd, config.n_embd)
        self.act_fun = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.act_fun(x))
        return self.drop(x)



class project_mlp_formula(nn.Module):
    def __init__(self, config):
        super(project_mlp_formula, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.fc2 = nn.Linear(2 * config.n_embd, config.n_embd)
        self.act_fun = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.act_fun(x))
        return self.drop(x)






class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, pointNetConfig=None):
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


        # self.soap_fc1 = nn.Linear(651525, 2 * embeddingSize)
        # self.soap_act = nn.GELU()
        # self.soap_fc2 = nn.Linear(2 * embeddingSize, embeddingSize)


        # input embedding stem
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


        # transformer
        # self.blocks_unmask = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.Trans_element_encoder = Transformer_formula_encoder(config)
        # self.block = Block(config)
        # decoder head
        self.ln_before_aggregate = nn.LayerNorm(config.n_embd)
        self.ln_after_aggregate = nn.LayerNorm(2 * config.n_embd)

        self.out1 = nn.Linear(2 * config.n_embd, 4 * config.n_embd)
        self.out_act = nn.ReLU()
        self.out2 = nn.Linear(4 * config.n_embd, 1)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
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
                elif 'adj_weight' in pn:  # 单独处理所有的 adj_weight
                    decay.add(fpn)  # 假设我们决定对所有的 adj_weight 应用权重衰减
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


        # forward the GPT model
        x = self.graph_convert_trans(nodes)  # each index maps to a (learnable) vector -> b x length x embedding
        graph = self.graph_embd(graph)
        mask = 1 - mask
        energy = energy.unsqueeze(0)
        # soap = self.soap_act(self.soap_fc1(soap))
        # soap = self.soap_fc2(soap)

        # x = torch.cat([soap.unsqueeze(1), x], axis=1)
        x = self.Trans_element_encoder(x, mask, edges, graph)



        x = self.ln_before_aggregate(x)

        ##############  aggregation  ###################
        # self.gru.flatten_parameters()
        x, _ = self.gru(self.ln_gru(x))
        att_score = self.att_score(x)
        mask = 1.0 - mask.float()
        mask = self.dropkey(mask)
        att_score = torch.where(mask==0.0, torch.ones_like(mask) * -1e10, att_score.squeeze(-1))
        att_weights = torch.softmax(att_score, dim=1)
        x = torch.sum(att_weights.unsqueeze(-1) * x, dim=1)

        

        # x = self.ln_after_aggregate(x)
        x = self.out_act(self.out1(x))
        pred = self.out2(x).squeeze()
        
        loss = nn.functional.mse_loss(pred.view(-1), energy.view(-1))
        return loss, pred
