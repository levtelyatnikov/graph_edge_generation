import os
import glob
import torch
import matplotlib.pyplot as plt
from PIL import Image
import sys
from PIL import Image
import wandb
from torchvision import utils as vutils
sys.path.append('/home/lev/')

class analize_dgm():
    def __init__(self):
        pass
        
   
    def temperature_stat(self, model):
        model = model.model
        self.d = {}
        for idx in range(len(model)):
            self.d[f"temperatue_{idx}"] =  model[idx].edge_conv.temperature.clone().detach()[0]
        
        

    def log_A_probs_masks(self, model, batch):
        p1d = (5,5,5,5)
        
        model = model.model
        batch.y = batch.y.cpu()
        idxs = torch.argsort(batch.y.cpu())

        # Adj matrix
        A = ((batch.y.unsqueeze(0) == batch.y.unsqueeze(1)) * 1)
        A = permute_CHW(A.unsqueeze(0), idx=idxs) * 0.5 # multiplization on 0.5 just to make it a bit gray

        A = torch.nn.functional.pad(A, p1d, mode='constant', value=0.7)
        out = [A]
        
        for idx in range(len(model)):
            probs = model[idx].edge_conv.probs.clone().detach().cpu()
            mask = model[idx].edge_conv.mask.clone().detach().cpu()

            probs = permute_CHW(probs.unsqueeze(0), idx=idxs)
            mask = permute_CHW(mask.unsqueeze(0), idx=idxs)

            
            probs = torch.nn.functional.pad(probs, p1d, mode='constant', value=0.7)
            mask = torch.nn.functional.pad(mask, p1d, mode='constant', value=0.7)
            

            out.extend([probs,
                        mask])

        out = torch.cat(out, dim=0).unsqueeze(1)
        images = vutils.make_grid(out.cpu(), normalize=False, nrow=1)
        images = torch.swapaxes(images, 2, 1)
        return images
        #elf.d['A_probs_masks'] = wandb.Image(images)
 

def permute_CHW(M, idx):
    M = M[:, idx, :]
    M = M[:, :, idx]
    return M
        
def cat_dicts(a, b):
    return dict(list(a.items()) + list(b.items()))

def check_if_file_exist(path): 
    return os.path.isfile(path) 



def clean_folder(folder):
    files = glob.glob(os.path.join(folder, '*'))
    for f in files:
        os.remove(f)
