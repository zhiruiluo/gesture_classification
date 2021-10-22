import argparse
import logging
import os
import time
import re
import json

from src.database.DatabaseManager import DatabaseManager
from src.dataloader.capg_dataloader import CapgDataLoader
from src.model.Tran import TRAN
from src.model.MSFCN_ATTN import get_msfcnattn
from src.trainer.base_trainer import (BaseTrainer, get_trainer,
                                      setup_earlystop, setup_modelcheckpoint)
from src.util.cuda_status import get_num_gpus


logger = logging.getLogger('trainer.MainTrainer')

def param_setting(args):
    return f"""ep={args.epochs}_nl={args.num_layers}_pt={args.patience}"""

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
    actual_stopped = early_stop.stopped_epoch if early_stop.stopped_epoch != 0 else args.epochs
    
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
    dl = CapgDataLoader(args.dataset, args.winsize, args.stride, batch_size=args.batch_size, nfold=args.nfold)
    dl.setup()
    args.nclass = dl.nclass
    
    if args.no_cuda:
        args.gpus = 0
    else:
        args.gpus = get_num_gpus()
    logger.info(f'args: {args}')
    
    # setup database
    model_name = args.model.split('_')[0]
    db_name = f'exp_{model_name}.db'
    c_entreis = customized_entries(args)
    db = DatabaseManager(db_name, c_entreis)

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
        r = get_custmoized_entry_dict(args,r)
        db.save_results(r)

        logger.info('test results {}'.format(test_result))
    
    exp_keys.pop('fold')
    logger.info('all results \n{}'.format(
            db.get_by_query_as_dataframe(exp_keys)
        ))

def add_model_hyperparameter(parser):
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers for LSTM')


def add_trainer_parameter(parser):
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=12,
                            help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--determ', action='store_true', default=False,
                        help='Deterministic flag')
    parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size')
    parser.add_argument('--nfold', type=int, default=10, help='nfold cross-validation', choices=[5,10])
    
def add_experiment_parameter(parser):
    parser.add_argument('--model', type=str, default='TRAN', choices=['TRAN','MSFCN_SEVT_P1','MSFCN_SEVT_P2',
                    'MSFCN_SEVT_S1','MSFCN_SEVT_S2'])
    parser.add_argument('--exp', type=int, default=1,
                            help='Unique experiment number')
    parser.add_argument('--dataset', type=str, default='CapgDataset', help='dataset ', 
                        choices=['CapgDataset'])
    parser.add_argument('--fold', type=int, default=0,
                            help='The fold number for current training')
    parser.add_argument('--nclass', type=int, default=2)

def add_customized_parameter(model, parser):
    model = model.split('_')[0]
    logger.info(model)
    if model in ['TRAN', 'MSFCN']:
        parser.add_argument('--winsize', type=int, default=50)
        parser.add_argument('--stride', type=int, default=10)

def customized_entries(args):
    model = args.model.split('_')[0]
    if model in ['TRAN', 'MSFCN']:
        c_entries = {
            'winsize': {'type':'integer', 'exist':'not null', 'key': True},
            'stride': {'type':'integer', 'exist':'not null','key':True},
        }

    write_customized_entries(args,c_entries)
    return c_entries

def write_customized_entries(args, c_entries):
    model = args.model.split('_')[0]
    save_to_path = os.path.join('./src/database/', f'Customized_TableEntries_{model}.json')
    if not os.path.isfile(save_to_path):
        with open(save_to_path,'w') as fp:
            json.dump(c_entries,fp,indent=4)

def get_custmoized_entry_dict(args, results_dict):
    args_dict = vars(args)
    c_entries = customized_entries(args)
    pairs = {}
    for k in c_entries:
        pairs[k] = args_dict[k]

    results_dict.update(pairs)
    return results_dict

def add_system_parameter(parser):
    parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--debug', action='store_true', default=False,
                            help='Debug flag')

def setup_arg():
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--model', type=str)
    model = model_parser.parse_known_args()[0].model

    parser = argparse.ArgumentParser()
    add_system_parameter(parser)
    add_model_hyperparameter(parser)
    add_experiment_parameter(parser)
    add_trainer_parameter(parser)
    add_customized_parameter(model, parser)
    return parser.parse_args()


class MainTrainer(BaseTrainer):
    def __init__(self,args,class_to_index):
        super().__init__(args.nclass)
        self.args = args
        self.class_to_index = class_to_index
        
        argu = args.model.split('_')
        if argu[0] == 'TRAN':
            self.model = TRAN(self.args.nclass,batch_size=self.args.batch_size,
                            pool=args.pool, num_layers=self.args.num_layers)
        elif argu[0] == 'MSFCN':
            self.model = get_msfcnattn(attn_layer=argu[1], attn_param={'structure':argu[2]},nclass=args.nclass,inplane=1,num_layers=args.num_layers)
        