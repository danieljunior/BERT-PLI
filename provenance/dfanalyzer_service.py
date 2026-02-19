from typing import List
import time
import pymonetdb
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element


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


############REMOVE BELOW####################
    def set_docs_pairs_generation_task(self, config, split):
        if self.bypass: 
            return

        input_file = config.get("data", split + "_coliee_file")
        output_file = (
            config.get("data", split + "_data_path")
            + "/"
            + config.get("data", split + "_file_list")
        )
        split_method = config.get("data", split + "_split_method")
        split_level = config.get("data", split + "_split_level")
        count = 0
        with open(output_file, "r") as f:
            count = sum(1 for line in f)

        t1 = Task(self.next_task_id(), self.dataflow_name, "docs_pairs_generation")
        t1_input = DataSet("coliee_file_input", [Element([input_file, split])])
        t1.add_dataset(t1_input)
        t1.begin()
        t1_output = DataSet(
            "splitted_doc_pair_dataset",
            [Element([output_file, split, count, split_method, split_level])],
        )
        t1.add_dataset(t1_output)
        t1.end()

        self.dependencies["docs_pairs_generation"] = t1

    def set_get_example_task(self, data):
        if self.bypass: 
            return

        docs1_elements = []
        docs2_elements = []
        labels_elements = []
        for temp in data:
            guid = temp["guid"]
            label = temp["label"]
            q_paras = temp["q_paras"]
            c_paras = temp["c_paras"]
            doc1 = guid.split("_")[0] + ".json"
            doc2 = guid.split("_")[1] + ".json"
            for c_idx, c_p in enumerate(c_paras):
                docs1_elements.append(
                    Element(
                        [
                            doc1,
                            c_p.replace("'", "''")
                            .encode("utf-8")
                            .decode("unicode-escape"),
                            c_idx,
                        ]
                    )
                )
            for q_idx, q_p in enumerate(q_paras):
                docs2_elements.append(
                    Element(
                        [
                            doc2,
                            q_p.replace("'", "''")
                            .encode("utf-8")
                            .decode("unicode-escape"),
                            q_idx,
                        ]
                    )
                )
            labels_elements.append(Element([label]))

        t_id = self.next_task_id()
        t2 = Task(
            t_id,
            self.dataflow_name,
            "get_doc1_example",
            dependency=self.dependencies["docs_pairs_generation"],
        )
        t2.begin()
        add_n_elements(t2, "doc1_segment", docs1_elements)
        t2.end()
        self.dependencies["get_doc1_example"] = t2

        t_id += 1
        t3 = Task(
            t_id,
            self.dataflow_name,
            "get_doc2_example",
            dependency=self.dependencies["docs_pairs_generation"],
        )
        t3.begin()
        add_n_elements(t3, "doc2_segment", docs2_elements)
        t3.end()
        self.dependencies["get_doc2_example"] = t3

        t_id += 1
        t4 = Task(
            t_id,
            self.dataflow_name,
            "get_label_example",
            dependency=self.dependencies["docs_pairs_generation"],
        )
        t4.begin()
        add_n_elements(t4, "label", labels_elements)
        t4.end()
        self.dependencies["get_label_example"] = t4

    def set_get_relevant_segments_task(self, data, criteria: str, epoch: int):
        if self.bypass: 
            return

        doc1_elements = []
        doc2_elements = []
        for row in data:
            q_file_id, c_file_id = row["guid"].split("_")
            if "c_selected_indices" in row and "q_selected_indices" in row:
                doc1_idx: List[int] = row["c_selected_indices"]
                doc2_idx: List[int] = row["q_selected_indices"]
                for idx in doc1_idx:
                    doc1_elements.append(Element([q_file_id + ".json", idx, criteria, epoch]))
                for idx in doc2_idx:
                    doc2_elements.append(Element([c_file_id + ".json", idx, criteria, epoch]))
            else: #no selection
                for idx, _ in enumerate(row["q_paras"]):
                    doc1_elements.append(Element([q_file_id + ".json", idx, criteria, epoch]))
                for idx, _ in enumerate(row["c_paras"]):
                    doc2_elements.append(Element([c_file_id + ".json", idx, criteria, epoch]))
        t_id = self.next_task_id()
        t5 = Task(
            t_id,
            self.dataflow_name,
            "doc1_relevant_segments_selection",
            dependency=self.dependencies["get_doc1_example"],
        )
        t5.begin()
        add_n_elements(t5, "doc1_relevant_segment", doc1_elements)
        t5.end()
        self.dependencies["doc1_relevant_segments_selection"] = t5
        
        t6 = Task(
            t_id + 1,
            self.dataflow_name,
            "doc2_relevant_segments_selection",
            dependency=self.dependencies["get_doc2_example"],
        )
        t6.begin()
        add_n_elements(t6, "doc2_relevant_segment", doc2_elements)
        t6.end()
        self.dependencies["doc2_relevant_segments_selection"] = t6
    
    def set_bert_scores_calculation(self, data, epoch: int):
        if self.bypass: 
            return

        t_id = self.next_task_id()
        # t7 = Task(t_id, self.dataflow_name, "bert_scores_calculation", 
        #           dependency=[self.dependencies["doc1_relevant_segments_selection"], 
        #                       self.dependencies["doc2_relevant_segments_selection"]])
        # t7.begin()
        # t7_output_elements = []
        # for qi, qrow in enumerate(data.get('original_lst')):
        #     for ci, scores in enumerate(qrow):
        #         t7_output_elements.append(
        #             Element([data.get('guid'), qi, ci, str(scores), epoch])
        #         )
        # add_n_elements(t7, "interaction_map", t7_output_elements)
        # t7.end()
        # self.dependencies["bert_scores_calculation"] = t7
        
        # Task 8: max_pooling
        # t8 = Task(t_id+1, self.dataflow_name, "max_pooling", dependency=t7)
        t8 = Task(t_id, self.dataflow_name, "max_pooling", 
                  dependency=[self.dependencies["doc1_relevant_segments_selection"], 
                              self.dependencies["doc2_relevant_segments_selection"]])
        t8.begin()
        t8_output_elements=[]
        for idx, value in zip(data.get('selected_c_indices'), data.get('max_out')):
            t8_output_elements.append(Element([data.get('guid'), str(idx), str(value), epoch]))
        
        add_n_elements(t8, "feature_vector", t8_output_elements)
        t8.end()
        self.dependencies["max_pooling"] = t8
    
    def set_classification_task(self, loss_metric, loss_value, predictions, epoch):
        if self.bypass: 
            return

        t_id = self.next_task_id()
        t9 = Task(t_id, self.dataflow_name, "classification", 
                  dependency=self.dependencies["max_pooling"])
        t9.begin()
        t9_output_elements = []
        for pred in predictions:
            guid, predicted, label = pred[0], pred[1], pred[2]
            t9_output_elements.append(Element([guid, predicted.item(),
                                               label.index(max(label)),
                                               loss_metric, loss_value, epoch]))
        add_n_elements(t9, "output", t9_output_elements)
        t9.end()
        self.dependencies["classification"] = t9
    
    def set_evaluation_task(self, eval_metrics: dict, epoch: int):
        if self.bypass: 
            return

        t_id = self.next_task_id()
        t10 = Task(t_id, self.dataflow_name, "evaluation", 
                   dependency=[self.dependencies["classification"],
                               self.dependencies["get_label_example"]])
        t10.begin()
        t10_output_elements = []
        for metric_name, metric_value in eval_metrics.items():
            t10_output_elements.append(Element([metric_name, metric_value, epoch]))
        add_n_elements(t10, "metric", t10_output_elements)
        t10.end()
        self.dependencies["evaluation"] = t10