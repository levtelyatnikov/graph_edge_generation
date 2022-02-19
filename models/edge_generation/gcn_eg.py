
from torch import nn
import torch
from omegaconf import DictConfig
from torch_geometric.nn import GCNConv
from torch_geometric.typing import  OptTensor
from torch.nn import Sequential as Seq, Linear, ReLU
from torch import Tensor
from models.edge_generation.bernulyEG import GumbleSigmoid
import numpy as np

class gcnConvEG_module(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(gcnConvEG_module, self).__init__()
        self.mapper = Seq(Linear(in_feat, in_feat),
                          ReLU(), 
                          Linear(in_feat, in_feat)
        )
        self.gumble = GumbleSigmoid()
        self.gcn = GCNConv(in_feat, out_feat)

        self.current_step = -1
        self.cfg = None

    def forward(self, x: Tensor, edge_index: OptTensor = None,
                                 edge_weight: OptTensor = None):
        x = self.mapper(x)

        if self.current_step < 1:
            self.setup_tau()

        self.tau = torch.tensor(self.warm_and_decay_lr_scheduler(), dtype=torch.float32).to(x.device)
        edge_index, edge_weight,\
        self.probs, self.mask, self.P = self.gumble.forward(x=x,
                                                            tau=self.tau)
        # to allow access to temperature for logger
        self.temperature = self.gumble.temperature
        
        x = self.gcn(x, edge_index, edge_weight)
        return x

    def setup_tau(self):
        
        self.total_steps = self.cfg.model.opt.max_epochs * self.cfg.model.opt.loader_batches 
        self.decay_steps = self.cfg.model.tau_params.decay_steps_pct * self.total_steps

    def warm_and_decay_lr_scheduler(self,):
        
        assert self.current_step <= self.total_steps
        factor = self.cfg.model.tau_params.scheduler_gamma ** (self.current_step / self.decay_steps)
        tau = max(factor * self.cfg.model.tau_params.initial_tau, self.cfg.model.tau_params.min)
        return [tau]
    

