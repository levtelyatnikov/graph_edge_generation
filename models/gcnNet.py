from torch import nn
from omegaconf import DictConfig
from torch_geometric.nn import Sequential, GCNConv

class GCNnet(nn.Module):
    """GCNnet"""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        
        self.cfg = cfg
        self.module = GCNConv
        if self.module is None:
           raise Exception('self.module is None, but have to be DynamicEdgeConv or DynamicEdgeConv_DCGv')

        # Build GCNnet

        # In size and out size depends on the dataset 
        cfg.in_channels = [cfg.insize] + list(cfg.in_channels)
        cfg.out_channels = list(cfg.out_channels) + [cfg.outsize]
        
        assert len(cfg.in_channels) == len(cfg.out_channels)

        sequential = []
        for idx, values in enumerate(zip(cfg.in_channels, cfg.out_channels)):
            in_feat, out_feat = values

            sequential.append((self.module(in_feat, out_feat), 'x, edge_index -> x'))

            if idx != len(cfg.in_channels) - 1:
                sequential.append(nn.ReLU())
            else: pass

        self.model = Sequential('x, edge_index', sequential)
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        logits = self.model(x=x, edge_index=edge_index)
        return logits

