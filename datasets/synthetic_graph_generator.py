import torch
import torch_cluster
from torch_geometric.data import Dataset, Data
from itertools import combinations
import numpy as np


class GraphGenerator():
    def __init__(self, features, labels, cfg):
        self.cfg = cfg
        self.features = features
        self.labels = labels

    def process(self,):
        if self.cfg.precompute_graph == "KNN": 
            return self.generate_knn_graph()

        elif self.cfg.precompute_graph == "random": 
            return self.random_graph()

        elif self.cfg.precompute_graph == "intraclass":
            return  self.random_intraclass_graph()

        elif self.cfg.precompute_graph == "intraclass_withnoise":
            return self.random_intraclass_withnoise_graph()

        elif self.cfg.precompute_graph == "fully_connected":
            return self.generate_fylly_connected()
        elif self.cfg.precompute_graph == "None": 
            return None

        else: Exception('Name of graph genetation has not been found')

    def generate_knn_graph(self,):
        assert self.cfg.precompute_graph == "KNN"
        assert self.cfg.precompute_graph_k < self.features.shape[0], f"k={self.cfg.precompute_graph_k} grater than datapoints {self.features.shape[0]}"
        assert self.cfg.n_batches == 1, 'Possible to work only with full graph in memory, NEED TO BE FIXED'
        
        data = Data(x=self.features)
        edge_index = torch_cluster.knn_graph(data.x, self.cfg.precompute_graph_k, batch=None, loop=True, flow='source_to_target').type(torch.LongTensor)
        
        print(f'KNN graph has been precomputed with k={self.cfg.precompute_graph_k}')
        return edge_index
    
    def random_graph(self,):
        assert self.cfg.precompute_graph == "random"
        assert self.cfg.precompute_graph_k <= self.features.shape[0], f"k={self.cfg.precompute_graph_k} grater than datapoints {self.features.shape[0]}"

        np.random.seed(self.cfg.graph_seed)
        n_nodes = self.features.shape[0]
        nodes = np.arange(n_nodes)
        n_neigh = self.cfg.precompute_graph_k

        

        children = self.generate_children(nodes, n_neigh).reshape(1,-1)
        parents = nodes.repeat(n_neigh).reshape(1,-1)
        print(f'RANDOM graph has been precomputed with k={self.cfg.precompute_graph_k}')
        return torch.tensor(np.concatenate([children, parents])).type(torch.LongTensor)

    def generate_children(self, nodes, n_neigh):
        n_nodes = nodes.shape[0]
        return np.array([np.random.choice(nodes, size=n_neigh, replace=False) for i in range(n_nodes)]).flatten()

    def random_intraclass_graph(self,):
        assert self.cfg.precompute_graph == "intraclass"
        
        n_neigh = self.cfg.precompute_graph_k
        unique = self.labels.unique()
        assert (torch.sum(self.labels.unsqueeze(0) == unique.unsqueeze(1), dim=1) >= n_neigh).all(), f"Some of unique labels={torch.sum(self.labels.unsqueeze(0) == unique.unsqueeze(1), dim=1)} smaller than k={n_neigh}"

        out_parents = []
        out_children = []
        for uni in unique:
            idx = torch.where(self.labels == uni)[0].cpu().numpy()

            out_children.extend(list(self.generate_children(idx, n_neigh)))
            out_parents.extend(list(idx.repeat(n_neigh)))

        children = np.array(out_children)
        parents = np.array(out_parents)
        
        assert children.shape == parents.shape
        sort_idx = np.argsort(parents)
        parents, children = parents[sort_idx].reshape(1,-1), children[sort_idx].reshape(1,-1)
        print(f'RANDOMINTRACLASS graph has been precomputed with k={self.cfg.precompute_graph_k}')
        return torch.tensor(np.concatenate([children, parents])).type(torch.LongTensor)
    
    def random_intraclass_withnoise_graph(self,):
        assert self.cfg.precompute_graph == "intraclass_withnoise"
        
        n_neigh = self.cfg.precompute_graph_k
        inter_n_neigh = self.cfg.inter_class_k
        unique = self.labels.unique()
        assert (torch.sum(self.labels.unsqueeze(0) == unique.unsqueeze(1), dim=1) >= n_neigh).all(), f"Some of unique labels={torch.sum(self.labels.unsqueeze(0) == unique.unsqueeze(1), dim=1)} smaller than k={n_neigh}"

        out_parents = []
        out_children = []
        for uni in unique:
            idx = torch.where(self.labels == uni)[0].cpu().numpy()

            idx_inter = torch.where(self.labels != uni)[0].cpu().numpy()

            out_children.extend(list(self.generate_children(idx, n_neigh)))
            out_parents.extend(list(idx.repeat(n_neigh)))

            out_children.extend(list(self.generate_children(idx_inter, inter_n_neigh)))
            out_parents.extend(list(idx_inter.repeat(inter_n_neigh)))

        children = np.array(out_children)
        parents = np.array(out_parents)
        #print(f"Number of clean edges: {len(children)}")
        assert children.shape == parents.shape

        # ========== ADD NOISE ==========
        # assert 1==0
        # nodes_idx = np.arange(n_neigh)
        # n_new_children = int(len(children) * self.cfg.noise_param)
        # noise_children = np.random.choice(nodes_idx, size=n_new_children, replace=True)
        # noise_parents = np.random.choice(nodes_idx, size=n_new_children, replace=True)

        # print(f"Number of noise clean edges: {len(noise_children)}")

        # parents = np.concatenate([parents, noise_parents])
        # children = np.concatenate([children, noise_children])

        print(f"Total number of edges: {len(children)}")
        # ================================

        assert children.shape == parents.shape

        sort_idx = np.argsort(parents)
        parents, children = parents[sort_idx].reshape(1,-1), children[sort_idx].reshape(1,-1)
        print(f'RANDOMINTRACLASS WITH NOISE graph has been precomputed with k={self.cfg.precompute_graph_k}')
        return torch.tensor(np.concatenate([children, parents])).type(torch.LongTensor)
    def generate_fylly_connected(self,):
        assert self.cfg.precompute_graph == "fully_connected"
        np.random.seed(self.cfg.graph_seed)
        
        n_nodes = self.features.shape[0]
        nodes = np.arange(n_nodes)
        
        edges = list(combinations(nodes, 2))

        parents = np.array([i[0] for i in edges])
        children = np.array([i[1] for i in edges])

        sort_idx = np.argsort(parents)
        parents, children = parents[sort_idx].reshape(1,-1), children[sort_idx].reshape(1,-1)
        print(f'FULLY CONNECTED graph has been precomputed with n_edges={len(parents)}')
        return torch.tensor(np.concatenate([children, parents])).type(torch.LongTensor)


        



