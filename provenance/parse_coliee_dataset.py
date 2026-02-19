import os
import time

from retrospective_service import RetrospectiveService
from prospective_service import ProspectiveService

dataflow_tag = os.getenv('DATAFLOW_TAG', ProspectiveService.DEFAULT_DATAFLOW_TAG)
provenance = RetrospectiveService(dataflow_tag)
input_data = {"coliee_dataset": [["/path/to/coliee/path", "train"]]}
with provenance.get_retrospective_data(ProspectiveService.TF_PARSE_COLIEE_DATASET, input_data) as result:
    time.sleep(2)
    result['coliee_parsed_dataset'] = [["/path/to/coliee_parsed/path", "train"]]
