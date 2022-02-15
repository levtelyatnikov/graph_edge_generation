from torch import nn
import torch
from omegaconf import DictConfig
from torch_geometric.nn import Sequential, GCNConv
from torch import Tensor
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConvEG(GCNConv):
    def __init__(self, in_feat, out_feat):
        super().__init__(in_feat, out_feat)
        self.temperature = torch.nn.Parameter(torch.Tensor([1]))
        self.eps = 10e-10

        
    def forward(self, x: Tensor, edge_index: OptTensor = None,
                                 edge_weight: OptTensor = None) -> Tensor: # edge_index: OptTensor = None, edge_weight: OptTensor = None
        """"""
        
        edge_index, probs, edge_weight, mask = self.sample_edges(x)

        self.edge_weight, self.edge_index = edge_weight, edge_index
        assert self.edge_weight.shape[0] == self.edge_index.shape[1]

        self.mask = mask
        self.probs = probs

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def sample_edges(self, nodes_emb):
        # Probabilities for edges
        probs = torch.exp(-self.temperature * torch.sum(nodes_emb - nodes_emb.unsqueeze(1), dim=-1) ** 2) + self.eps
        #probs = torch.log(probs + self.eps)
        P = self.gumbel_sigmoid_adj(probs)

        # Get parent child list
        childer = torch.arange(P.shape[0]).repeat(P.shape[1]).to(nodes_emb.device)
        parents = torch.arange(P.shape[0]).view(-1,1).repeat((1,P.shape[1])).flatten().to(nodes_emb.device)
        edge_index = torch.stack([childer, parents])

        # Get weight sequence of 0, 1 for edge_list
        mask = torch.clamp((P + P.T), min=0 , max=1)
        edge_weights = mask.view((-1, ))
        
        return edge_index, probs, edge_weights, mask

    def gumbel_sigmoid_adj(self, logits, tau: float = 10, hard: bool = True):
        
        gumbels = (
            -torch.empty_like(
                logits,
                memory_format=torch.legacy_contiguous_format
                ).exponential_().log() # ~Gumbel(0,1)
            )
        gumbels = (logits + gumbels) / tau  
        y_soft = gumbels.sigmoid()

        if hard:
            y_hard = (y_soft>=0.5)*1
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

















