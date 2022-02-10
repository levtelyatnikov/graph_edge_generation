from torch import nn
from omegaconf import DictConfig
from models.edge_conv import DynamicEdgeConv, DynamicEdgeConv_DGM, DynamicEdgeConvMINE

class EdgeNet(nn.Module):
    """EdgeNet"""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        dynamic_edge_type = {
            "DynamicEdgeConv": DynamicEdgeConv,
            "DynamicEdgeConv_DGM": DynamicEdgeConv_DGM,
            "DynamicEdgeConvMINE": DynamicEdgeConvMINE
            }
        self.cfg = cfg
        self.module = dynamic_edge_type.get(cfg.edge_generation_type)
        if self.module is None:
           raise Exception('self.module is None, but have to be DynamicEdgeConv or DynamicEdgeConv_DCGv')

        # Build EdgeNet
        modules = []

        # In size and out size depends on the dataset 
        cfg.in_channels = [cfg.insize] + list(cfg.in_channels)
        cfg.out_channels = list(cfg.out_channels) + [cfg.outsize]
        
        assert len(cfg.in_channels) == len(cfg.out_channels)
        assert len(cfg.in_channels) == len(cfg.k)

        for idx, values in enumerate(zip(cfg.in_channels, cfg.out_channels, cfg.k)):
            in_feat, out_feat, k = values
            sequential = nn.Sequential()

            sequential.add_module(f"edge_conv", self.module(in_feat, out_feat, k=k))

            if idx != len(cfg.in_channels) - 1:
                sequential.add_module(f"act", nn.ReLU())
            else: pass
            modules.append(sequential)

        self.model = nn.Sequential(*modules)
        
    def forward(self, batch):
        x = batch.x
        logits = self.model(x)
        return logits

