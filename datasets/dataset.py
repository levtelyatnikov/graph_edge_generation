import sys
import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, Data
from sklearn.datasets import load_breast_cancer, load_digits, load_boston, load_iris, load_diabetes, load_wine

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

        data = eval(cfg.dataset_name)
        self.features = torch.tensor(data['data']).type(torch.float32)
        self.labels = torch.tensor(data['target']).type(torch.int64)

        self.split = split

        self.graph_size = cfg.graph_size
        self.val_size = cfg.val_size
        self.seed = cfg.data_seed
        self.n_batches = cfg.n_batches

        self.setup()

    def setup(self):
        # features_labels = list(zip(self.features, self.labels))

        # Split
        X_train, X_val, y_train, y_val = train_test_split(self.features, self.labels,
                                test_size=self.val_size,
                                random_state=self.seed)

        if self.split == "train":
            self.features = X_train
            self.labels = y_train
        elif self.split == "val":
            self.graph_size = 'None'
            self.features = X_val
            self.labels =y_val
        else:
            print("Specify dataset split correctly", file=sys.stderr)
        
        
        self.idxs = np.arange(self.features.shape[0])

    def get(self, idx):
        """Return image and label"""
        if self.graph_size == "None":
            idxs = self.idxs
        else:
            idxs = np.random.choice(self.idxs, size=self.graph_size)
       
        features = self.features[idxs]
        label = self.labels[idxs]
        
        return Data(x=features, y=label)

    def len(self):
        return self.n_batches
    
    
