from logging import raiseExceptions
import sys
import numpy as np
import torch
import pandas as pd
import torch_cluster
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, Data
from datasets.synthetic_graph_generator import GraphGenerator

from datasets.preprocess_dataset import upload_data
from sklearn.datasets import load_breast_cancer, load_digits, load_boston, load_iris, load_diabetes, load_wine
from ml_flow.ml_pipline import ml_flow

class CustomDataset(Dataset):
    """Dataset example"""
    def __init__(self, split, cfg: DictConfig):
        super().__init__(root='None', transform=None, pre_transform=None, pre_filter=None)
        """Initialize

        cfg:
         :data_dir: data directory
         :transforms: TransformObject Class which is defined in the transformfactory.py file
         :target_transforms: TransformLabel Class which is defined in the transformfactory.py file

         :split: train/val split
         :val_size: validation size
         :seed: seed
        """

        
        self.cfg = cfg
        self.split = split
        self.graph_size = cfg.graph_size
        self.val_size = cfg.val_size
        self.seed = cfg.data_seed
        self.n_batches = cfg.n_batches

        self.setup()

    def setup(self):
        # Upload data
        self.features, self.labels = upload_data(self.cfg.dataset_name)
        assert self.features.shape[0]>0 and  self.labels.shape[0]>0
        

        # Obtain train test split
        self.features, self.labels = self.train_test_split()
        print(f'{self.split} set shape', self.features.shape)

        # Get edges_index if needed
        self.edge_index = self.get_graph()
        
        
    def get_graph(self,):
        gg = GraphGenerator(self.features, self.labels, self.cfg)
        return gg.process()
        
        
    def train_test_split(self,):
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(self.features, self.labels,
                                                            test_size=self.val_size,
                                                            random_state=self.seed)

        self.f1_svm, self.acc_svm, self.f1_lin, self.acc_lin = ml_flow(X_train, X_val, y_train, y_val)
        

        if self.split == "train":
            features = X_train
            labels = y_train
        elif self.split == "val":
            self.graph_size = 'None'
            features = X_val
            labels =y_val
        else:
            print("Specify dataset split correctly", file=sys.stderr)

        self.idxs = np.arange(features.shape[0])
        return features, labels
        

    def get(self, idx):
        """Return image and label"""
        if self.graph_size == "None":
            idxs = self.idxs
        else:
            idxs = np.random.choice(self.idxs, size=self.graph_size)
       
        features = self.features[idxs]
        label = self.labels[idxs]
        data = Data(x=features, y=label)

        if self.cfg.precompute_graph != 'None':
            data.edge_index = self.edge_index 

        return data
        

    def len(self):
        return self.n_batches
    
    
