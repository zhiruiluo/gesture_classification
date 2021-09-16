import sqlite3 as lite
import os
from typing import Dict, List
import pandas as pd
import json
import logging
logger = logging.getLogger("database.DatabaseManager")


class DatabaseManager():
    def __init__(self, db_name, customized_entries=None):
        self.db_name = db_name
        self.customized_entries = customized_entries
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.db_path = os.path.join(self.path,self.db_name)
        self.timeout = 10000

        self.register_type()
        self.setup_database()

    def register_type(self):
        lite.register_adapter(bool,int)
        lite.register_converter('bool',lambda v: int(v) != 0)

    def _read_json(self):
        path = os.path.join(self.path,'basic_table.json')
        with open(path, 'r') as jf:
            datatable = json.loads(jf.read())
        
        return datatable

    def _update_entry(self, datatable):
        table_entries = [(k,v['type'],v['exist']) for k,v in datatable['tableEntry'].items()]
        columns = [k for k in datatable['tableEntry'].keys()]
        keys = [k for k,v in datatable['tableEntry'].items() if v['key']]

        if self.customized_entries:
            last_key = 0
            for i, entry in enumerate(datatable['tableEntry'].items()):
                name, attribute =entry
                if not attribute['key']:
                    last_key = i
                    break
            
            for name, entry in self.customized_entries.items():
                if entry['key']:
                    columns.insert(last_key, name)
                    keys.insert(last_key,name)
                    table_entries.insert(last_key, (name,entry['type'],entry['exist']))
                    last_key+=1
                else:
                    columns.append(name)
                    keys.append(name)
                    table_entries.append((name,entry['type'],entry['exist']))

        return columns, keys, table_entries

    def setup_database(self):
        #self.create_database()
        datatable = self._read_json()
        columns, keys, table_entries = self._update_entry(datatable)

        self.table_name = datatable['tableName']
        self.columns = columns
        self.keys = keys
        self.table_entries = table_entries
        
        create_tb_sql = self.build_create_table_sql(self.table_name,self.table_entries,self.keys)
        self.create_database(create_tb_sql)

    def build_create_table_sql(self, table_name,table_entries,keys):
        tb_entry = []
        # add columns to the table
        for c in table_entries:
            tb_entry.append(' '.join(filter(None,c)))
        # add primary keys to the table
        tb_entry.append('primary key (' + ', '.join(keys) + ')')
        # concatenate tb entry
        tb_entry_str = ',\n'.join(tb_entry)
        # generate create table 
        create_tb_sql = """PRAGMA foreign_keys = OFF;\n""" + \
                        """CREATE TABLE {} (\n""".format(table_name) + \
                        tb_entry_str + \
                        """\n);"""

        return create_tb_sql

    def create_database(self,create_tb_sql):
        if not os.path.isfile(self.db_path):
            #os.system(f'sqlite3 {os.path.join(self.path,db_name)} ";"')
            self.create_table(create_tb_sql)

    def connect(self):
        conn = lite.connect(self.db_path,detect_types=lite.PARSE_DECLTYPES)
        return conn

    def create_table(self, create_table_sql):
        logger.info('[database] create table')
        
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.executescript(create_table_sql)
            conn.commit()
        except lite.Error as e:
            logger.error('[create_table] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()

    def save_results(self, results: Dict):
        logger.info('[database] save results')
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % int(self.timeout))
            cur.execute('begin')

            insert_str = f"""insert into {self.table_name} ({','.join(self.columns)}) \n""" + \
                         f"""values ({','.join(['?' for i in range(len(self.columns))])}) \n""" + \
                         f"""on conflict({','.join(self.keys)}) \n""" + \
                         f"""do update \n""" +  \
                         f"""   set acc=excluded.acc \n""" + \
                         f"""       where acc < excluded.acc;"""
            
            cur.execute(insert_str, 
                        ([results[c] for c in self.columns])
                    )
            conn.commit()
            logger.info(f'[save_results] successful\n {results}')
        except lite.Error as e:
            logger.error('[save_results] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
    
    def save_status(self, p: Dict):
        logger.info('[database] save_status')
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % int(self.timeout))
            cur.execute('begin')

            insert_str = f"""insert into {self.table_name} ({','.join(self.columns)}) \n""" + \
                         f"""values ({','.join(['?' for i in range(len(self.columns))])}) \n""" + \
                         f"""on conflict({','.join(self.keys)}) \n""" + \
                         f"""do update \n""" +  \
                         f"""   set acc=excluded.acc \n""" + \
                         f"""       where acc < excluded.acc;"""
            cur.execute(insert_str, 
                        ([p[c] for c in self.columns])
                    )

            conn.commit()
        except lite.Error as e:
            logger.error('[save_status] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
    
    def read_status(self, p: Dict):
        res = None
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            
            select_str = f"""select * from {self.table_name} where {' and '.join([k+'=?' for k in self.keys])};"""
            cur.execute(select_str,
                ([p[c] for c in self.keys])
            )
            
            res = cur.fetchone()
        except lite.Error as e:
            logger.error('[read_status] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
        
        return res

    def check_finished(self, p: Dict):
        res = self.read_status(p)
        if res == None:
            return False
        return True
        
    def read_all(self) -> List[lite.Row]:
        res = None
        conn = None
        try:
            conn = self.connect()
            conn.row_factory = lite.Row
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            
            select_str = f"""select * from {self.table_name}\n""" + \
                         f"""order by {','.join(self.keys)};"""
            cur.execute(select_str)
            
            res = cur.fetchall()
        except lite.Error as e:
            logger.error('[read_status] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
        
        return res

    def read_by_keys(self, query):
        res = None
        conn = None
        try:
            conn = self.connect()
            conn.row_factory = lite.Row
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            
            select_str = f"""select * from {self.table_name}\n""" + \
                         f"""where { ' and '.join([k+'=?' for k in query.keys()]) }\n""" + \
                         f"""order by {','.join(self.keys)};"""

            cur.execute(select_str,
                tuple(query.values())
            )
            
            res = cur.fetchall()
        except lite.Error as e:
            logger.error('[read_status] error {}'.format(e.args[0]))

        finally:
            if conn:
                conn.close()
        
        return res

    def get_by_query_as_dataframe(self, query):
        res = self.read_by_keys(query)
        if not res:
            return None

        df = pd.DataFrame(res, columns=res[0].keys())
        return df

    def get_all_as_dataframe(self):
        res = self.read_all()
        if not res:
            return None
        
        df = pd.DataFrame(res, columns=res[0].keys())
        return df

    def delete_all_entry(self):
        conn = None
        is_success = True
        answer = input(f'Do you want to delete all results in {self.db_path} for table {self.table_name}? yes/no\n')
        if answer == 'yes':
            pass
        else:
            logger.info('exit without deleting')
            return
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            
            select_str = f"""delete from {self.table_name};"""
            cur.execute(select_str)
            
            conn.commit()
            logger.info(f'records are deleted successfully.')
        except lite.Error as e:
            logger.error('[read_status] error {}'.format(e.args[0]))
            is_success = False
        finally:
            if conn:
                conn.close()
        
        return is_success


def _test():
    db_name = 'exp_test1.db'
    c_entries = {
        'stride': {'type':'integer', 'exist':'not null','key':True},
        'acc1': {'type':'integer','exist':'','key':False}
    }
    #db = DatabaseManager(db_name=db_name, customized_entries=c_entries)
    db = DatabaseManager(db_name=db_name)
    reaults = { k:'1' for k in db.columns}
    db.save_results(reaults)
    print(dict(db.read_all()[0]))

if __name__ == '__main__':
    _test()

