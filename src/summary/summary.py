import logging
import os
import json
from numpy.lib.npyio import save

import pandas as pd
from src.database.DatabaseManager import DatabaseManager

logger = logging.getLogger('src.summary.summary')



class Summary():
    def __init__(self, db_name, model=None):
        self.db_name = db_name
        self.model = model
        
        self.db = DatabaseManager(db_name=db_name, customized_entries=self.load_customized_entries())

    def load_customized_entries(self):
        if not self.model:
            return None

        saved_path = os.path.join('./src/database/', f'Customized_TableEntries_{self.model}.json')
        with open(saved_path, 'r') as fp:
            c_entries = json.load(fp)
        return c_entries

    def read_database(self):
        results = self.db.get_all_as_dataframe()
        results = self.sort_keys(results)
        logger.info(f'print database {self.db_name} \n {results}')
        return results

    def sort_keys(self, results: pd.DataFrame):
        k = self.db.keys.copy()
        k.remove('fold')
        k.append('fold')
        col = k + self.db.nonkeys.copy()
        results = results[col]
        results = results.sort_values(by=k).reset_index(drop=True)
        return results

    def get_average_folds(self):
        results = self.db.get_average_folds()
        logger.info(f'average results over nfold\n{results}')

    def savedata(self):
        results = self.read_database()
        if not os.path.isdir('results/'):
            os.mkdir('results/')
        if results is not None:
            results.to_csv(f'results/database={self.db_name}.csv')
        else:
            logger.info(f'[Savedata] Empty results table for database {self.db_name}')
