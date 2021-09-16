import logging
from typing import Any, List

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import torch


logger = logging.getLogger('training.base_trainer')

def setup_modelcheckpoint(monitor, filename, mode):
    ckp_cb = ModelCheckpoint(dirpath='model_checkpoint',
            filename=filename + '-{epoch:02d}-{val_acc_epoch:.3f}',
            monitor=monitor,
            save_top_k=1,
            mode=mode
            )
    return ckp_cb

def setup_logger(exp_name,version=None):
    pl_logger = TensorBoardLogger(
        save_dir='tensorboard_logs',
        name=exp_name,
        version=version,
        )
    return pl_logger

def setup_earlystop(monitor,patience,mode):
    earlystop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode
    )
    return earlystop

def get_trainer(gpus, epochs, earlystop, ckp, exp_name, version=None):
    trainer = pl.Trainer(
        gpus=gpus,
        #auto_select_gpus = True,
        max_epochs=epochs,
        progress_bar_refresh_rate=0.5,
        flush_logs_every_n_steps=100,
        logger=setup_logger(exp_name, version),
        callbacks=[earlystop, ckp],
    )
    return trainer


class BaseTrainer(pl.LightningModule):
    def __init__(self):
        super(BaseTrainer,self).__init__()


    def forward(self, batch):
        pass 

    def loss(self, pred, label):
        pass

    def configure_optimizers(self):
        pass

    def train_metrics(self, phase, pred, label):
        pass

    def val_metrics(self, phase, pred, label):
        pass

    def test_metrics(self, phase, pred, label):
        pass

    def train_metrics_end(self, phase):
        pass

    def val_metrics_end(self, phase):
        pass

    def test_metrics_end(self, phase):
        pass

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, batch[1])
        
        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        if phase == 'train':
            self.train_metrics(phase, y_hat, batch[1])
        elif phase == 'val':
            self.val_metrics(phase, y_hat, batch[1])
        elif phase == 'test':
            self.test_metrics(phase, y_hat, batch[1])
        
        return loss


    def training_step(self, batch, batch_nb):
        phase = 'train'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def training_epoch_end(self, outputs) -> None:
        phase = 'train'
        self.train_metrics_end(phase)

    def validation_step(self, batch, batch_nb):
        phase = 'val'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'val'
        self.val_metrics_end(phase)

    def test_step(self, batch, batch_nb):
        phase = 'test'
        # fwd
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, batch[1])
        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics(phase, y_hat, batch[1])
        
        return 

    def test_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'test'
        self.test_metrics_end(phase)
        
    def get_early_stop(self, patience):
        return setup_earlystop('val_acc_epoch', patience, 'max')

    def get_checkpoint_callback(self, filename):
        return setup_modelcheckpoint('val_acc_epoch', filename, 'max')