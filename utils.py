import torch
from pytorch_lightning import Callback
import wandb

class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images_train, images_val = pl_module.sample_images()
                if images_train != None:
                    trainer.logger.experiment.log({"images_train": [wandb.Image(images_train)]}, commit=False)
                    trainer.logger.experiment.log({"images_val": [wandb.Image(images_val)]}, commit=False)
                
                
