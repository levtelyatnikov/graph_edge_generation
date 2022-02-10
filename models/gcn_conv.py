import torch
import torch_cluster
from torch_geometric.nn import MessagePassing,GCNConv
from torch.nn import Sequential as Seq, Linear, ReLU



class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, k):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
        

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
       
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels] 
        # Workout the case when k=0, hence one node
        # if torch.all(x_j == x_i):
        #     return self.mlp_k0(x_i)

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
    
class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k):
        super().__init__(in_channels, out_channels, k)
        self.k_degree = k
        
    def forward(self, x, batch=None):
        # Case when k = 1, hence only self-loop
        if self.k_degree==0:
            edge_index = torch_cluster.knn_graph(x, 1, batch=None, loop=True, flow='source_to_target') # return self correspondence [[0, 1, 2,], [0, 1, 2]]
        elif self.k_degree>=1:
            edge_index = torch_cluster.knn_graph(x, self.k_degree, batch=None, loop=False, flow='source_to_target')
        else: print('Ups...')
    
        self.edge_index = edge_index
        return super().forward(x, edge_index)
