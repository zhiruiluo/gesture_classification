import logging
from src.database.DatabaseManager import DatabaseManager
import os

logger = logging.getLogger('src.summary.summary')



class Summary():
    def __init__(self, db_name):
        self.db = DatabaseManager(db_name=db_name)
        self.db_name = db_name

    def read_database(self):
        results = self.db.get_all_as_dataframe()
        logger.info(f'print database {self.db_name} \n {results}')
        return results
    
    def savedata(self):
        results = self.read_database()
        if not os.path.isdir('results/'):
            os.mkdir('results/')
        if results is not None:
            results.to_csv(f'results/database={self.db_name}.csv')
        else:
            logger.info(f'[Savedata] Empty results table for database {self.db_name}')