import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig
from models.EdgeNet import EdgeNet
from torchmetrics import Accuracy, AveragePrecision, AUROC
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np


class Network(nn.Module):
    """Network"""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        networks = {'EdgeNet':EdgeNet}
        # Import model
        self.model = networks.get(cfg.type)(cfg) # cfg.model.type
        
        # Loss
        self.crossentropy = torch.nn.CrossEntropyLoss()

        # Metrics
        self.acc = Accuracy()
        self.avr_precision = AveragePrecision(num_classes = cfg.outsize)
        self.aucroc = AUROC(num_classes = cfg.outsize)

    def forward(self, x):
        
        logits = self.model(x)
        return logits
    
    def loss_function(self, batch):
        """Loss fuction

        This function implements all logic during train step. In this way you
        model class is selfc ontained, hence there isn't need to change code
        in method.py when model is substituted with other one.
        """
        
        x, y = batch.x.squeeze(0), batch.y.squeeze(0)
        logits = self.forward(x)  
        loss_CE = self.crossentropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)
        f1 = f1_score(preds.cpu(), y.cpu(), average='weighted')

        probs = F.softmax(logits, dim=1)
        avr_precision = self.avr_precision(probs, y)
        aucroc = self.aucroc(probs, y)

        if self.cfg.edge_generation_type == "DynamicEdgeConv_DGM":
            res_dict_dgm = self.temperature_stat()
            prob_loss = self.prob_loss(y=y, preds=preds, device=x.device)
            res_dict_dgm['loss_prob'] = self.cfg.prob_reg * prob_loss
            loss = loss_CE + self.cfg.prob_reg * prob_loss
        else: 
            loss = loss_CE

        res_dict = {
            "loss": loss,
            "cross_entrop":loss_CE, 
            "accuracy": acc,
            "f1": torch.Tensor([f1]),
            "avr_precision": avr_precision,
            "aucroc": aucroc,
            
            
            }
        if self.cfg.edge_generation_type == "DynamicEdgeConv_DGM":
            res_dict = cat_dicts(res_dict, res_dict_dgm)
        
        return res_dict
        
    def temperature_stat(self):
        d = {}
        for idx in range(len(self.model.model)):
            d[f"temperatue_{idx}"] =  self.model.model[idx].edge_conv.temperature.clone().detach()[0]
        return d
    
    def delta_vector(self, y, preds, device):
        class_acc = self.class_accuracy(y, preds)
        y, preds = y.view(1, -1), preds.view(1, -1)
        
        # get accuracy weight for each class in sequence 
        class_acc = class_acc[y].to(device)
        
        # 1) (y == preds) * (class_acc - 1) is equivalent to acc_cl - 1
        # 2) (y != preds) * class_acc) is equivalent to acc_cl
        # Hence from 1) y == preds we have acc_cl - 1;  
        # From 2) y != preds we have acc_cl
        delta_vector = (y != preds) * class_acc  + (y == preds) * (class_acc - 1)
        return delta_vector

    def prob_loss(self, y, preds, device):
        eps = 10e-6
        module = self.model.model
        n_modules = len(module)
        prob_L = torch.cat([module[i].edge_conv.probs.unsqueeze(0) for i in range(n_modules)], dim=0)
        mask_L = torch.cat([module[i].edge_conv.mask.unsqueeze(0) for i in range(n_modules)], dim=0)

        d_vec = self.delta_vector(y=y, preds=preds, device=device)

        weight_mask = (prob_L * mask_L) * d_vec 
        #negative_mask = (mask_L == 0 ) * 1
        #weight_mask = weight_mask + negative_mask
        #weight_mask = torch.prod(weight_mask, dim=1)
        
        return torch.sum(weight_mask) #weight_mask.prod(dim=0).sum().type(torch.FloatTensor)

    def class_accuracy(self, y, preds):
        cm = confusion_matrix(y.cpu(), preds.cpu())
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = torch.tensor(cm.diagonal()) 
        return class_acc

def cat_dicts(a, b):
    return dict(list(a.items()) + list(b.items()))