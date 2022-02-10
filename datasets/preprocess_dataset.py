import glob
import pandas as pd
import torch
import numpy as np

def upload_data(dataset_name):
    datsets_csv = pd.read_csv('/home/lev/object-centric/edge-generation/configs/datasets.cvs', sep='\t', index_col=0)
    if dataset_name in ['load_breast_cancer', 'load_digits', 'load_boston', 'load_iris', 'load_diabetes', 'load_wine']:
        data = eval(dataset_name)
        features = data['data']
        labels = data['target']

    elif dataset_name in list(datsets_csv.name):
        df = datsets_csv[datsets_csv.name == dataset_name]
        assert df.shape[0] == 1, "The dataset name is not unique"
        data = pd.read_table(df.path.iloc[0], index_col=0)

        # n_initial = data.shape[0]
        # data = data.groupby('clase').filter(lambda x : len(x)/n_initial > 0.05)
        # data.reset_index(inplace=True, drop=True)
        

        labels = np.array(data['clase'])
        features = np.array(data.drop(columns=['clase']))

        print("Class-counts: \n", data.clase.value_counts())

        # Doublecheck
        assert data.shape[1] - 1 == features.shape[1]
    else: pass

    return torch.tensor(features).type(torch.float32), torch.tensor(labels).type(torch.int64)


# datsets_csv = pd.read_csv('/home/lev/object-centric/edge-generation/configs/datasets.cvs', sep='\t', index_col=0)
# if cfg.dataset_name in ['load_breast_cancer', 'load_digits', 'load_boston', 'load_iris', 'load_diabetes', 'load_wine']:
#     data = eval(cfg.dataset_name)
#     self.features = torch.tensor(data['data']).type(torch.float32)
#     self.labels = torch.tensor(data['target']).type(torch.int64)

# else:
#     X, y = upload_data(cfg.dataset_path)
#     self.features = torch.tensor(X).type(torch.float32)
#     self.labels = torch.tensor(y).type(torch.int64)