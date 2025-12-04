
from typing import List
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
import provenance.bert_pli_prospective as prospective
from .bert_pli_retrospective import add_n_elements
class DfanalyzerService:

    def __init__(self, dataflow_name: str, shift_task_id: int = 0):
        self.dataflow_name = dataflow_name
        self.shift_task_id = shift_task_id
        self.dataflow = None
        self.dependencies = {}

    def create_dataflow(self):
        self.dataflow = prospective.create_bert_pli_dataflow(dataflow_tag=self.dataflow_name)
        self.dataflow.save()

    def set_docs_pairs_generation_task(self, config, split):
        input_file = config.get("data", split + "_coliee_file")
        output_file = config.get("data", split + "_data_path") + "/" + \
                        config.get("data", split + "_file_list")
        split_method = config.get("data", split + "_split_method")
        split_level = config.get("data", split + "_split_level")
        count = 0
        with open(output_file, 'r') as f:
            count = sum(1 for line in f)

        t1 = Task(1 + self.shift_task_id, self.dataflow_name, "docs_pairs_generation")
        t1_input = DataSet("coliee_file_input", [Element([input_file, split])])
        t1.add_dataset(t1_input)
        t1.begin()
        t1_output = DataSet("splitted_doc_pair_dataset",
                            [Element([output_file, split, count, split_method, split_level])])        
        t1.add_dataset(t1_output)
        t1.end()
        
        self.dependencies["docs_pairs_generation"] = t1
    
    def set_get_example_task(self, data):
        docs1_elements = []
        docs2_elements = []
        labels_elements = []
        for temp in data:
            guid = temp['guid']
            label = temp['label']
            q_paras = temp['q_paras']
            c_paras = temp['c_paras']
            doc1 = guid.split("_")[0]+".json"
            doc2 = guid.split("_")[1]+".json"
            for c_idx, c_p in enumerate(c_paras):
                docs1_elements.append(Element([doc1,
                                               c_p.replace("'", "''").encode('utf-8').decode('unicode-escape'),
                                               c_idx,]))
            for q_idx, q_p in enumerate(q_paras):
                docs2_elements.append(Element([doc2,
                                               q_p.replace("'", "''").encode('utf-8').decode('unicode-escape'),
                                               q_idx,]))
            labels_elements.append(Element([label]))

        t_id = 2 + self.shift_task_id
        t2 = Task(t_id, self.dataflow_name, "get_doc1_example", 
                  dependency=self.dependencies["docs_pairs_generation"])
        t2.begin()
        add_n_elements(t2, "doc1_segment", docs1_elements)
        t2.end()
        self.dependencies["get_doc1_example"] = t2

        t_id += 1
        t3 = Task(t_id, self.dataflow_name, "get_doc2_example", 
                  dependency=self.dependencies["docs_pairs_generation"])
        t3.begin()
        add_n_elements(t3, "doc2_segment", docs2_elements)
        t3.end()
        self.dependencies["get_doc2_example"] = t3
        
        t_id += 1
        t4 = Task(t_id, self.dataflow_name, "get_label_example", 
                  dependency=self.dependencies["docs_pairs_generation"])
        t4.begin()
        add_n_elements(t4, "label", labels_elements)
        t4.end()
        self.dependencies["get_label_example"] = t4