import time
import pymonetdb


class DfanalyzerService:
    URL = "dfanalyzer"
    PORT = 50000
    DATABASE = "dataflow_analyzer"
    USERNAME = "monetdb"
    PASSWORD = "monetdb"

    def __init__(self, bypass: bool = False):
        self.bypass = bypass
    
    def get_monet_connection(self):
        conn = pymonetdb.connect(
            hostname=self.URL,
            port=self.PORT,
            database=self.DATABASE,
            username=self.USERNAME,
            password=self.PASSWORD,
        )
        return conn

    def dataflow_exists(self, dataflow_name : str) -> bool:
        if self.bypass: 
            return False

        conn = self.get_monet_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM dataflow WHERE tag = %s;", (dataflow_name,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        return row is not None

    def get_last_task_id(self, df_tag: str) -> int:
        if self.bypass: 
            return

        conn = self.get_monet_connection()
        conn.commit()
        cursor = conn.cursor()
        query_1 = """
            SELECT t.identifier 
            FROM task t
            ORDER BY t.identifier DESC 
            LIMIT 1;
        """
        cursor.execute(query_1)
        row = cursor.fetchone()

        # If no tasks exist, return 0
        if row is None:
            last_identifier = 0
        else:
            last_identifier = row[0]

        cursor.close()
        conn.close()

        return last_identifier

    def get_last_task_id_from_dataflow(self, df_tag: str) -> int:
        if self.bypass: 
            return

        conn = self.get_monet_connection()
        cursor = conn.cursor()

        # 1. Get df_id
        cursor.execute("SELECT id FROM dataflow WHERE tag = %s;", (df_tag,))
        row = cursor.fetchone()

        if row is None:
            cursor.close()
            conn.close()
            raise ValueError(f"No dataflow found with tag='{df_tag}'")

        df_id = row[0]

        # 2. Get latest task identifier (descending order, take first)
        query_1 = """
            SELECT t.identifier 
            FROM task t
            INNER JOIN dataflow_version dv ON t.df_version = dv.version
            WHERE dv.df_id = %s
            ORDER BY dv.version DESC, t.identifier DESC 
            LIMIT 1;
        """
        cursor.execute(query_1, (df_id,))
        row = cursor.fetchone()

        # If no tasks exist, return 0
        if row is None:
            last_identifier = 0
        else:
            last_identifier = row[0]

        cursor.close()
        conn.close()

        return last_identifier

    def next_task_id(self, dataflow_tag : str) -> int:
        if self.bypass: 
            return

        last_id = self.get_last_task_id(dataflow_tag)
        return last_id + 1

    def update_custom_text_columns(self):
        conn = self.get_monet_connection()
        cursor = conn.cursor()
        already = False

        while not already:
            conn.commit()
            cursor.execute("SELECT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ds_entailment_config' AND system = false);")
            response = cursor.fetchone()[0]
            if response:
                already = True
            else:
                time.sleep(1)

        changes = [
            ['ds_entailment_config', 'config' ],
            ['ds_poolout_config', 'config' ],
            ['ds_test_config', 'config' ],
            ['ds_classifier_config', 'config' ],
        ]

        query = """
            ALTER TABLE %s ADD COLUMN %s varchar(5000);
            UPDATE %s SET %s = CONVERT(%s, varchar(5000));
            DROP VIEW %s restrict;
            ALTER TABLE %s DROP COLUMN %s restrict;
            ALTER TABLE %s RENAME COLUMN %s TO %s;
            CREATE VIEW %s AS SELECT * FROM %s;  
        """
        for change in changes:
            ds_table = change[0]
            column = change[1]
            tmp_column = 'tmp_'+ column
            view = ds_table.split('ds_')[1]
            cursor.execute(query % ( ds_table, tmp_column, 
                                    ds_table, tmp_column, column,
                                    view,
                                    ds_table, column,
                                    ds_table, tmp_column, column,
                                    view, ds_table,
                                )
                            )
            conn.commit()
        cursor.close()
        conn.close()
