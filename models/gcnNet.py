from torch import nn
import torch
from omegaconf import DictConfig
from torch_geometric.nn import Sequential, GCNConv
from models.GCNConvEdgeGeneration import GCNConvEG
from collections import OrderedDict
class GCNnet(nn.Module):
    """GCNnet"""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        dynamic_edge_type = {
            "GCNConv": GCNConv,
            "GCNConvEG": GCNConvEG,
            }
        
        self.cfg = cfg
        
        self.module = dynamic_edge_type.get(cfg.edge_generation_type)
        
        if self.module is None:
           raise Exception('self.module is None, but have to be DynamicEdgeConv or DynamicEdgeConv_DCGv')

        # Build GCNnet

        # In size and out size depends on the dataset 
        cfg.in_channels = [cfg.insize] + list(cfg.in_channels)
        cfg.out_channels = list(cfg.out_channels) + [cfg.outsize]
        
        assert len(cfg.in_channels) == len(cfg.out_channels)

        #sequential = []
        names = []
        modules = []
        for idx, values in enumerate(zip(cfg.in_channels, cfg.out_channels)):
            in_feat, out_feat = values

            names.append(f'gcn_eg{idx}')
            modules.append((self.module(in_feat, out_feat), 'x, edge_index, edge_weight -> x'))

            if idx != len(cfg.in_channels) - 1:
                names.append(f'act{idx}')
                modules.append(nn.ReLU())
            else: pass

        self.model = Sequential('x, edge_index, edge_weight', OrderedDict(zip(names, modules)))
        
    def forward(self, batch):
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_weight
        logits = self.model(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return logits

# # ---------------------------------- GCNnet_edgeGeneration ------------------------------------

# class GCNnet_edgeGeneration(GCNnet):
#     def __init__(self, cfg: DictConfig):
#         super().__init__(cfg)

#         # initialize temp. parameter with 1, however not sure this is a good idea :)
#         self.temperature = torch.nn.Parameter(torch.Tensor([0.001]))
    
#     def forward(self, batch):
        
#         edge_index, probs, edge_weights = self.sample_edges(batch.x)
#         batch.edge_weight = edge_weights
#         super.forward()


#     def sample_edges(self, nodes_emb):
#         # Probabilities for edges
#         probs = torch.exp(-self.temperature * torch.sum(nodes_emb - nodes_emb.unsqueeze(1), dim=-1) ** 2) + self.eps
       
#         P = self.gumbel_sigmoid_adj(probs)
#         # Get parent child list
#         childer = torch.arange(P.shape[0]).repeat(P.shape[1])
#         parents = torch.arange(P.shape[0]).view(-1,1).repeat((1,P.shape[1])).flatten()
#         edge_index = torch.stack([childer, parents])

#         # Get weight sequence of 0, 1 for edge_list
#         mask = torch.clamp((P + P.T), min=0 , max=1)
#         edge_weights = mask.view((-1, ))
        
#         return edge_index, probs, edge_weights

#     def gumbel_sigmoid_adj(logits, tau: float = 1, hard: bool = True):
#         gumbels = (
#             -torch.empty_like(
#                 logits,
#                 memory_format=torch.legacy_contiguous_format
#                 ).exponential_().log() # ~Gumbel(0,1)
#             )
#         gumbels = (logits + gumbels) / tau  
#         y_soft = gumbels.sigmoid()

#         if hard:
#             y_hard = (y_soft>=0.5)*1
            
#             ret = y_hard - y_soft.detach() + y_soft
#         else:
#             ret = y_soft
#         return ret
