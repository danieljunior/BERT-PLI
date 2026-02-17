import pickle
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from persistence_service import PersistenceService
from prospective_service import ProspectiveService

prov_persistence = PersistenceService(ProspectiveService.DEFAULT_DATAFLOW_TAG)
tf_dependencies_tasks = prov_persistence.load_task_dependencies(ProspectiveService.TF_PARSE_COLIEE_DATASET)

parse_coliee_dataset = Task(2, ProspectiveService.DEFAULT_DATAFLOW_TAG, 
                            ProspectiveService.TF_PARSE_COLIEE_DATASET,
                            dependency=tf_dependencies_tasks)
coliee_dataset = DataSet("coliee_dataset", [Element(["/path/to/coliee/path","train"])])
parse_coliee_dataset.add_dataset(coliee_dataset)
parse_coliee_dataset.begin()
coliee_parsed_dataset = DataSet("coliee_parsed_dataset", [Element(["/path/to/coliee_parsed/path","train"])])
parse_coliee_dataset.add_dataset(coliee_parsed_dataset)
parse_coliee_dataset.end()

prov_persistence.save_task(ProspectiveService.TF_PARSE_COLIEE_DATASET, parse_coliee_dataset)
