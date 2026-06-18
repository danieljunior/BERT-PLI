import pickle
from pathlib import Path
from typing import Any, Dict, Optional, List
from dfa_lib_python.task import Task
from dfa_lib_python.dataflow import Dataflow

class PersistenceService:
    """Service for managing transformation and dependency provenance objects."""
    DEFAULT_STORAGE_DIR = '/app/provenance/storage/'

    def __init__(self, dataflow_storage_dir: str = 'dataflow') -> None:
        """
        Initialize the provenance service.
        
        Args:
            storage_dir: Directory where provenance files will be stored
        """
        self.storage_dir = Path(self.DEFAULT_STORAGE_DIR + dataflow_storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dependencies_file = self.storage_dir / 'dependencies.bin'
        self.dependencies: Dict[str, Any] = self._load_dependencies()
    
    def _load_dependencies(self) -> Dict[str, Any]:
        """Load the dependencies registry from disk."""
        try:
            with open(self.dependencies_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_dependencies(self) -> None:
        """Save the dependencies registry to disk."""
        with open(self.dependencies_file, 'wb') as f:
            pickle.dump(self.dependencies, f)
    
    def save_task(self, task_name: str, task: Task) -> str:
        """
        Save a task provenance object.
        
        Args:
            task_name: Unique identifier for the task
            task: Task object to save
            
        Returns:
            Path to the saved task file
        """
        task_file = self.storage_dir / f'{task_name}.bin'
        with open(task_file, 'wb') as f:
            pickle.dump(task, f)
        
        self.dependencies[task_name] = str(task_file)
        self._save_dependencies()
        
        return str(task_file)
    
    def load_task(self, task_name: str) -> Optional[Task]:
        """
        Load a task provenance object.
        
        Args:
            task_name: Unique identifier for the task
            
        Returns:
            Loaded Task object or None if not found
        """
        if task_name not in self.dependencies:
            return None
        
        task_file = self.dependencies[task_name]
        try:
            with open(task_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
    
    def load_dependencies(self, *task_names: str) -> Dict[str, Task]:
        """
        Load multiple task dependencies.
        
        Args:
            *task_names: Variable number of task names to load
            
        Returns:
            Dictionary mapping task names to loaded Task objects
        """
        loaded = {}
        for name in task_names:
            task = self.load_task(name)
            if task is not None:
                loaded[name] = task
        return loaded

    def load_task_dependencies(self, dataflow, task_name: str) -> Dict[str, Task]:
        """
        Load all dependencies for a given task.
        
        Args:
            task_name: Name of the task to load dependencies for
        """
        tf = next(tf for tf in dataflow.transformations 
                       if tf["tag"] == task_name)
        tf_dependencies = [ dp 
                            for dp 
                            in [t.get('dependency') for t in tf['sets']] 
                            if dp is not None]
        
        return [self.load_task(dp) for dp in tf_dependencies]
    
    def has_task(self, task_name: str) -> bool:
        """Check if a task exists in the registry."""
        return task_name in self.dependencies
    
    def list_tasks(self) -> List[str]:
        """List all registered tasks."""
        return list(self.dependencies.keys())
    
    def clear_dependencies(self) -> None:
        """Clear all dependencies."""
        self.dependencies = {}
        self._save_dependencies()
    
    def save_dataflow(self, dataflow_tag: str, dataflow: Dataflow) -> str:
        """
        Save a Dataflow object.
        
        Args:
            dataflow_tag: Unique identifier for the dataflow
            dataflow: Dataflow object to save
            
        Returns:
            Path to the saved dataflow file
        """
        dataflow_file = self.storage_dir / f'{dataflow_tag}.bin'
        with open(dataflow_file, 'wb') as f:
            pickle.dump(dataflow, f)
        
        return str(dataflow_file)
    
    def load_dataflow(self, dataflow_tag: str) -> Optional[Dataflow]:
        """
        Load a Dataflow object.
        
        Args:
            dataflow_tag: Unique identifier for the dataflow
            
        Returns:
            Loaded Dataflow object or None if not found
        """
        dataflow_file = self.storage_dir / f'{dataflow_tag}.bin'
        try:
            with open(dataflow_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataflow with tag '{dataflow_tag}' not found in storage.")