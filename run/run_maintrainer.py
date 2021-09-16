import os
import sys
from pathlib import Path
cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
import logging
import time

def setup_logger(model):
    logger = logging.getLogger()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not os.path.isdir('logging'):
        os.mkdir('logging')

    jobid = os.environ.get('SLURM_JOB_ID')

    if jobid is not None:
        filehandler = logging.FileHandler(f"logging/{model}_{jobid}_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}.log")
    else:
        filehandler = logging.FileHandler(f"logging/{model}_{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}.log")
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger

def test():
    from src.trainer.main_trainer import train, setup_arg
    args = setup_arg()
    logger = setup_logger(args.model)

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    train(args)
    

if __name__ == '__main__':
    test()