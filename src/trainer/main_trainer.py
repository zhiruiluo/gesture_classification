import argparse
import logging
import os
import time
import re

import torch.nn.functional as F
from src.database.DatabaseManager import DatabaseManager
from src.dataloader.capg_dataloader import CapgDataLoader
from src.model.Tran import TRAN
from src.trainer.base_trainer import (BaseTrainer, get_trainer,
                                      setup_earlystop, setup_modelcheckpoint)
from src.util.cuda_status import get_num_gpus
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import F1, Accuracy

logger = logging.getLogger('trainer.MainTrainer')

def param_setting(args):
    return f"""ep={args.epochs}_h={args.height}_nl={args.num_layers}_pt={args.patience}"""

def get_exp_keys(keys, args):
    return {k: vars(args)[k] for k in keys}

def unique_name(exp_keys):
    #return f"""exp={exp_keys['exp']}_nclass={exp_keys['nclass']}_model={exp_keys['model']}_ds={exp_keys['dataset']}_fold={exp_keys['fold']}"""
    return "-".join([f"{k}={v}" for k,v in exp_keys.items()])

def parse_best_epoch(best_model_path):
    name = os.path.basename(best_model_path)
    ret = {}
    for pair in re.split('-',name):
        if "=" in pair:
            k, v = pair.split("=")
            if k == 'epoch':
                ret[k] = v
            elif k == 'val_acc_epoch':
                ret[k] = v.strip('.ckpt')
                break

    return ret

def get_version_name(args):
    jobid = os.environ.get('SLURM_JOB_ID')
    if jobid is not None:
        version = f'{jobid}_fold_{args.fold}_{time.strftime("%h-%d-%Y-%H:%M:%S")}'
    else:
        version = f'fold_{args.fold}_{time.strftime("%h-%d-%Y-%H:%M:%S")}'
    return version

def test_results_to_results(args, exp_keys, test_result, confmx, early_stop, best_info, label):
    actual_stopped = args.epochs if early_stop.stopped_epoch else early_stop.stopped_epoch
    
    r = dict(
        exp = exp_keys['exp'],
        nclass = exp_keys['nclass'],
        model = exp_keys['model'],
        dataset = exp_keys['dataset'],
        fold = exp_keys['fold'],
        acc = test_result['test_acc'],
        f1micro = test_result['test_f1micro'],
        f1macro = test_result['test_f1macro'],
        stopepoch = actual_stopped,
        bestepoch = best_info['epoch'],
        bestvalacc = best_info['val_acc_epoch'],
        confmx = str(confmx),
        label = label,
        time = time.strftime('%h/%d/%Y-%H:%M:%S')
    )
    logger.info(f'results {r}')
    return r

def train(args):
    args.determ = True
    dl = CapgDataLoader(args.dataset, batch_size=args.batch_size, nfold=args.nfold)
    dl.setup()
    args.nclass = dl.nclass
    args.nc = dl.nc
    
    if args.no_cuda:
        args.gpus = 0
    else:
        args.gpus = get_num_gpus()
    logger.info(f'args: {args}')
    
    # setup database
    db_name = f'exp_{args.model}.db'
    db = DatabaseManager(db_name)

    exp_keys = get_exp_keys(db.keys, args)
    if not db.check_finished(exp_keys):
        logger.info(f'setup training environment')
        # setup lighting model 
        model = MainTrainer(args, dl.class_to_index)
        # setup early-stop callback
        early_stop = model.get_early_stop(patience=args.patience)
        # setup checkpoint callback
        ckp_cb = model.get_checkpoint_callback(unique_name(exp_keys))
        # setup pytorch-lightning trainer
        trainer = get_trainer(args.gpus, args.epochs, early_stop, ckp_cb, unique_name(exp_keys), get_version_name(args))
        
        # get ith-fold of the training data
        dl.get_fold(args.fold)
        # fit model
        trainer.fit(model,datamodule=dl)
        # test model
        test_result = trainer.test(ckpt_path=ckp_cb.best_model_path, datamodule=dl)[0]
        
        ## convert test_result dictionary to dictionary
        logger.info(f'best mode path {ckp_cb.best_model_path}')
        r = test_results_to_results(args, exp_keys, test_result, model.get_test_confmx(), early_stop, parse_best_epoch(ckp_cb.best_model_path), param_setting(args))
        db.save_results(r)

        logger.info('test results {}'.format(test_result))
    
    exp_keys.pop('fold')
    logger.info('all results \n{}'.format(
            db.get_by_query_as_dataframe(exp_keys)
        ))

def setup_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=12,
                            help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--determ', action='store_true', default=False,
                        help='Deterministic flag')
    parser.add_argument('--width', type=int, default=128,
                        help='Width of CNN input')
    parser.add_argument('--height', type=int, default=128,
                        help='Height of CNN input')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers for LSTM')
    parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size')
    parser.add_argument('--debug', action='store_true', default=False,
                            help='Debug flag')
    parser.add_argument('--exp', type=int, default=1,
                            help='Unique experiment number')
    parser.add_argument('--fold', type=int, default=0,
                            help='The fold number for current training')
    parser.add_argument('--model', type=str, default='TRAN', choices=['TRAN'])
    parser.add_argument('--dataset', type=str, default='CapgDataset', help='dataset ', 
                        choices=['CapgDataset'])
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--nfold', type=int, default=10, help='nfold cross-validation', choices=[5,10])
    return parser.parse_args()


class MainTrainer(BaseTrainer):
    def __init__(self,args,class_to_index):
        super().__init__()
        self.args = args
        self.class_to_index = class_to_index
        
        if args.model == 'TRAN':
            self.model = TRAN(self.args.nclass,self.args.nc,
                            batch_size=self.args.batch_size, pool=args.pool, num_layers=self.args.num_layers)
        else:
            pass

        # train metrics
        self.train_acc_score = Accuracy()
        self.train_f1_score_macro = F1(num_classes=args.nclass,average='macro')
        self.train_f1_score_micro = F1(num_classes=args.nclass,average='micro')
        self.train_confmx = ConfusionMatrix(args.nclass)

        # val metrics
        self.val_acc_score = Accuracy()
        self.val_f1_score_macro = F1(num_classes=args.nclass,average='macro')
        self.val_f1_score_micro = F1(num_classes=args.nclass,average='micro')
        self.val_confmx = ConfusionMatrix(args.nclass)

        # test metrics
        self.test_acc_score = Accuracy()
        self.test_f1_score_macro = F1(num_classes=args.nclass,average='macro')
        self.test_f1_score_micro = F1(num_classes=args.nclass,average='micro')
        self.test_confmx = ConfusionMatrix(args.nclass)
        
        self.stored_test_confmx = None

    def get_test_confmx(self):
        if self.stored_test_confmx is not None:
            return self.stored_test_confmx.cpu().numpy().tolist()
        return []

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
    
    def train_metrics(self, phase, pred, label):
        acc = self.train_acc_score(pred, label)
        self.train_f1_score_micro(pred, label)
        self.train_f1_score_macro(pred, label)
        self.train_confmx(pred, label)
        self.log(f'{phase}_acc_step', acc, sync_dist=True, prog_bar=True)
    
    def val_metrics(self, phase, pred, label):
        acc = self.val_acc_score(pred, label)
        self.val_f1_score_micro(pred, label)
        self.val_f1_score_macro(pred, label)
        self.val_confmx(pred, label)
        self.log(f'{phase}_acc_step', acc, sync_dist=True, prog_bar=True)

    def test_metrics(self, phase, pred, label):
        acc = self.test_acc_score(pred, label)
        self.test_f1_score_micro(pred, label)
        self.test_f1_score_macro(pred, label)
        self.test_confmx(pred, label)
        self.log(f'{phase}_acc_step', acc, sync_dist=True, prog_bar=True)

    def train_metrics_end(self, phase):
        metrics = {}
        metrics['acc'] = self.train_acc_score.compute()
        metrics['f1micro'] = self.train_f1_score_micro.compute()
        metrics['f1macro'] = self.train_f1_score_macro.compute()
        metrics['confmx'] = self.train_confmx.compute()
        
        self.log_epoch_end(phase, metrics)
        
        self.train_acc_score.reset()
        self.train_f1_score_micro.reset()
        self.train_f1_score_macro.reset()
        self.train_confmx.reset()

    def val_metrics_end(self, phase):
        metrics = {}
        metrics['acc'] = self.val_acc_score.compute()
        metrics['f1micro'] = self.val_f1_score_micro.compute()
        metrics['f1macro'] = self.val_f1_score_macro.compute()
        metrics['confmx'] = self.val_confmx.compute()
        
        self.log_epoch_end(phase, metrics)
        
        self.val_acc_score.reset()
        self.val_f1_score_micro.reset()
        self.val_f1_score_macro.reset()
        self.val_confmx.reset()
    
    def test_metrics_end(self, phase):
        metrics = {}
        metrics['acc'] = self.test_acc_score.compute()
        metrics['f1micro'] = self.test_f1_score_micro.compute()
        metrics['f1macro'] = self.test_f1_score_macro.compute()
        metrics['confmx'] = self.test_confmx.compute()
        
        self.log_epoch_end(phase, metrics)
        self.stored_test_confmx = metrics['confmx']

    def log_epoch_end(self, phase, metrics):
        self.log(f'{phase}_acc_epoch', metrics['acc'])
        self.log(f'{phase}_f1micro', metrics['f1micro'])
        self.log(f'{phase}_f1macro', metrics['f1macro'])
        self.log(f'{phase}_acc', metrics['acc'])
        

        logger.info(f'[{phase}_acc_epoch] {metrics["acc"]} at {self.current_epoch}')
        logger.info(f'[{phase}_f1_score] {metrics["f1micro"]}')
        logger.info(f'[{phase}_f1_score_macro] {metrics["f1macro"]}')
        logger.info(f'[{phase}_confmx] \n {metrics["confmx"]}')

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

