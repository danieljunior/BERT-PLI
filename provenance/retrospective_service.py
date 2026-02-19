import logging
from contextlib import contextmanager

from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from persistence_service import PersistenceService
from prospective_service import ProspectiveService
from dfanalyzer_service import DfanalyzerService

# Configure logger with custom formatter
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - (DFANALYZER) - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class RetrospectiveService:
    def __init__(self, dataflow_tag : str):
        self.dataflow_tag = dataflow_tag
        self.dfanalyzer_service = DfanalyzerService()
        self.persistence_service = PersistenceService()
        
        if not self.dfanalyzer_service.dataflow_exists(self.dataflow_tag):
            self.prospective_service = ProspectiveService(self.dataflow_tag, self.persistence_service)
            self.dataflow = self.prospective_service.build_dataflow()
        else:
            self.dataflow = self.persistence_service.load_dataflow(self.dataflow_tag)

    @contextmanager
    def get_retrospective_data(self, task_name:str, input_data:dict):
        """Context manager for retrospective data retrieval"""
        try:
            # LOG INPUT DATA RECORDING
            logger.info(f"[{task_name}] Starting task with input data: {input_data}")
            
            task_dependencies = self.persistence_service.load_task_dependencies(self.dataflow, task_name)

            self.task = Task(self.dfanalyzer_service.next_task_id(self.dataflow_tag), 
                        self.dataflow_tag, task_name,
                        dependency=task_dependencies)
            # input_data must be in the form of a dictionary where keys are dataset names and 
            # values are the dataset values
            # Ex.: {"dataset1": [[1,2,3]], "dataset2": [["a", "b", "c"]]}
            for dt_name, dt_values in input_data.items():
                for elem in dt_values:
                    input_dataset = DataSet(dt_name, [Element(elem)])
                    self.task.add_dataset(input_dataset)
                    self.task.save()
                logger.debug(f"[{task_name}] Added input dataset '{dt_name}': {dt_values}")

            self.task.begin()
            logger.info(f"[{task_name}] Task execution started")
            # result must be populated by the caller with the output datasets in the form of a dictionary where keys are 
            # dataset names and values are the dataset values
            # Ex.: {"output_dataset1": [[4,5,6]], "output_dataset2": [["d", "e", "f"]]}
            self.result = {}
            
            # LOG CALLER EXECUTION
            logger.info(f"[{task_name}] Yielding control to caller for execution")
            yield self.result
            logger.info(f"[{task_name}] Caller execution completed with output: {self.result}")

        except Exception as e:
            logger.error(f"[{task_name}] Error during task execution: {str(e)}", exc_info=True)
            raise
            
        finally:
            # LOG OUTPUT DATA RECORDING
            logger.info(f"[{task_name}] Recording {len(self.result)} output dataset(s)")
            
            for dt_name, dt_values in self.result.items():
                for elem in dt_values:
                    output_dataset = DataSet(dt_name, [Element(elem)])
                    self.task.add_dataset(output_dataset)
                    self.task.save()
                logger.debug(f"[{task_name}] Added output dataset '{dt_name}': {dt_values}")
            
            self.task.end()
            self.persistence_service.save_task(task_name, self.task)
            logger.info(f"[{task_name}] Task completed and saved successfully")
            self.task = None
            self.result = None