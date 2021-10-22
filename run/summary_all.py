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
    db_names = {}
    for fn in os.listdir('src/database/'):
        if fn.endswith('.db') and fn.startswith('exp') and 'test' not in fn and 'grid' not in fn:
            
            model = fn.replace('.db','').split('_')[1]
            db_names[fn] = model
    print(db_names)
    for db_name, model in db_names.items():
        s = Summary(db_name, model)
        s.savedata()
        s.get_average_folds()
