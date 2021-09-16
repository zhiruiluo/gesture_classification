import os
import sys
from pathlib import Path

cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))
import logging

from src.summary.summary import Summary

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info('[summary_all]')

if __name__ == '__main__':
    db_names = ['exp_TRAN.db']
    for db_name in db_names:
        s = Summary(db_name)
        s.savedata()
