import os
import sys
from pathlib import Path
cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(cur_path.parent))

import logging
import time

logger = logging.getLogger()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

from src.dataloader.capg_dataloader import CapgDataLoader

if __name__ == '__main__':
    dl = CapgDataLoader()
    dl.setup()
    dl.get_fold(0)
    print(dl.class_to_index)
    for x , y in dl.train_dataloader:
        pass
        # print(y)