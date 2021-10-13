import logging
from typing import Any, List

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import F1, Accuracy
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
        progress_bar_refresh_rate=0.1,
        flush_logs_every_n_steps=100,
        logger=setup_logger(exp_name, version),
        callbacks=[earlystop, ckp],
    )
    return trainer


class BaseTrainer(pl.LightningModule):
    def __init__(self, nclass):
        super(BaseTrainer,self).__init__()

        self.all_metrics = nn.ModuleDict()
        for phase in ['train','val', 'test']:
            self.all_metrics[phase+'_metrics'] = nn.ModuleDict({
                    "acc": Accuracy(),
                    "f1macro": F1(num_classes=nclass,average='macro'),
                    "f1micro": F1(num_classes=nclass,average='micro'),
                    "confmx": ConfusionMatrix(nclass)
                })

        self.stored_test_confmx = None

    def get_test_confmx(self):
        if self.stored_test_confmx is not None:
            return self.stored_test_confmx.cpu().numpy().tolist()
        return []

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler1 = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps=7, num_training_steps=self.args.epochs)
        #scheduler = CosineAnnealingLR(optimizer, self.args.epochs)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': scheduler1
        }


    def forward(self, batch):
        X, y = batch
        #batch, seq_len = X.size()
        #X = X.view(batch,1,seq_len)
        logger.debug('model {} {}'.format(X[0].shape, y[0].shape))
        y_hat = self.model(X)
        return y_hat

    def loss(self, pred, label):
        logger.debug('loss {} {}'.format(pred, label))
        loss = F.cross_entropy(pred, label)
        return loss
    
    def metrics(self, phase, pred, label):
        phase_metrics = self.all_metrics[phase+'_metrics']
        for mk, metric in phase_metrics.items():
            result = metric(pred,label)
            if mk == 'acc':
                self.log(f'{phase}_acc_step', result, sync_dist=True, prog_bar=True)

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase+'_metrics']
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute()
            metric.reset()

        self.log_epoch_end(phase, metrics)
        if phase == 'test':
            self.stored_test_confmx = metrics['confmx']

    def log_epoch_end(self, phase, metrics):
        self.log(f'{phase}_acc_epoch', metrics['acc'])
        self.log(f'{phase}_f1micro', metrics['f1micro'])
        self.log(f'{phase}_f1macro', metrics['f1macro'])
        self.log(f'{phase}_acc', metrics['acc'])
        
        logger.info(f'[{phase}_acc_epoch] {metrics["acc"]} at {self.current_epoch}')
        logger.info(f'[{phase}_f1micro] {metrics["f1micro"]}')
        logger.info(f'[{phase}_f1macro] {metrics["f1macro"]}')
        logger.info(f'[{phase}_confmx] \n {torch.tensor(metrics["confmx"],dtype=torch.long)}')

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, batch[1])
        
        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        self.metrics(phase, y_hat, batch[1])
        return loss


    def training_step(self, batch, batch_nb):
        phase = 'train'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def training_epoch_end(self, outputs) -> None:
        phase = 'train'
        self.metrics_end(phase)

    def validation_step(self, batch, batch_nb):
        phase = 'val'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'val'
        self.metrics_end(phase)

    def test_step(self, batch, batch_nb):
        phase = 'test'
        # fwd
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, batch[1])
        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics(phase, y_hat, batch[1])
        
        return 

    def test_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'test'
        self.metrics_end(phase)
        
    def get_early_stop(self, patience):
        return setup_earlystop('val_acc_epoch', patience, 'max')

    def get_checkpoint_callback(self, filename):
        return setup_modelcheckpoint('val_acc_epoch', filename, 'max')