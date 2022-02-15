import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from models.model import Network




class LitModel(pl.LightningModule):
    def __init__(self, datamodule, cfg: DictConfig):
        super().__init__()
        pl.utilities.seed.seed_everything(cfg.seed_everything)
        # -----------------!!!----------------
        # If you want to substitute models from .yaml file
        # it is necessary to map models name: class
        
        # -----------------!!!----------------
        
        # save pytorch lightning parameters   
        # this row makes ur parameters be available with self.hparams name
        self.save_hyperparameters(cfg)

        # get model from .yaml file
        self.model = Network(cfg.dataloader.f1_svm,
                             cfg.dataloader.acc_svm,
                             cfg)

        self.datamodule = datamodule
       
    # logic for a single training step
    def training_step(self, batch, batch_idx):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        val_loss = self.model.loss_function(batch)
        return val_loss
        
    def validation_epoch_end(self, outputs):
        logs = {}
        keys = outputs[0].keys()
        for key in keys:
            logs["val_" + key] = torch.stack([x[key] for x in outputs]).mean()
        
        self.log_dict(logs, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.model.opt.lr, weight_decay=self.hparams.model.opt.weight_decay)

        warmup_steps_pct = self.hparams.model.opt.warmup_steps_pct
        decay_steps_pct = self.hparams.model.opt.decay_steps_pct
        total_steps = self.hparams.model.opt.max_epochs * self.hparams.model.opt.loader_batches #len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step <= total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.hparams.model.opt.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )

    def sample_images(self):
        if self.hparams.model.edge_generation_type == 'DynamicEdgeConv_DGM' \
            or self.hparams.model.edge_generation_type == 'GCNConvEG':
            batch_val = next(iter(self.datamodule.val_dataloader()))
            batch_train = next(iter(self.datamodule.train_dataloader()))

            with torch.no_grad():
                self.model.eval()
                self.model(batch_train.to(self.device))
            images_train = self.model.analyze_stat.log_A_probs_masks(model=self.model.model, batch=batch_train)

            with torch.no_grad():
                self.model.eval()
                self.model(batch_val.to(self.device))
            images_val = self.model.analyze_stat.log_A_probs_masks(model=self.model.model, batch=batch_val)
            
            return images_train, images_val
        else: 
            return None, None 
