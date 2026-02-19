import os
import time

from retrospective_service import RetrospectiveService
from prospective_service import ProspectiveService

dataflow_tag = os.getenv('DATAFLOW_TAG', ProspectiveService.DEFAULT_DATAFLOW_TAG)
provenance = RetrospectiveService(dataflow_tag)
input_data = {"poolout_config": [["config","finetuned_bert_checkpoint"]]}


with provenance.get_retrospective_data(ProspectiveService.TF_POOLOUT, input_data) as result:
    time.sleep(2)
    result['poolout_data'] = [["1","poolout_file","selected_sentences_file"],
                              ["2","poolout_file","selected_sentences_file"]]

