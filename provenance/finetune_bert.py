from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
import pickle

from persistence_service import PersistenceService
from prospective_service import ProspectiveService

prov_persistence = PersistenceService(ProspectiveService.DEFAULT_DATAFLOW_TAG)
tf_dependencies_tasks = prov_persistence.load_task_dependencies(ProspectiveService.TF_FINETUNE_BERT)

finetune_bert = Task(1, ProspectiveService.DEFAULT_DATAFLOW_TAG, 
                        ProspectiveService.TF_FINETUNE_BERT,
                        dependency=tf_dependencies_tasks)
bert_base = DataSet("bert_base", [Element(["/path/to/bert_base/checkpoint"])])
finetune_bert.add_dataset(bert_base)
entailment_config = DataSet("entailment_config", [Element(["/path/to/entailment/config"])])
finetune_bert.add_dataset(entailment_config)
finetune_bert.begin()
finetuned_bert = DataSet("finetuned_bert_model", [Element(["/path/to/finetuned_bert/last_checkpoint"])])
finetune_bert.add_dataset(finetuned_bert)
finetune_bert.end()

prov_persistence.save_task(ProspectiveService.TF_FINETUNE_BERT, finetune_bert)
