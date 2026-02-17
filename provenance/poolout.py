import pickle
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element

from persistence_service import PersistenceService
from prospective_service import ProspectiveService

prov_persistence = PersistenceService(ProspectiveService.DEFAULT_DATAFLOW_TAG)
tf_dependencies_tasks = prov_persistence.load_task_dependencies(ProspectiveService.TF_POOLOUT)

poolout = Task(3 , ProspectiveService.DEFAULT_DATAFLOW_TAG, 
                    ProspectiveService.TF_POOLOUT,
                    dependency=tf_dependencies_tasks)
coliee_dataset = DataSet("poolout_config", [Element(["blablabla"])])
poolout.add_dataset(coliee_dataset)
poolout.begin()
poolout_data = DataSet("poolout_data", [Element(["poolout_file","selected_sentences_file"])])
poolout.add_dataset(poolout_data)
poolout.end()
prov_persistence.save_task(ProspectiveService.TF_POOLOUT, poolout)
