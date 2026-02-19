import os
import time

from retrospective_service import RetrospectiveService
from prospective_service import ProspectiveService

dataflow_tag = os.getenv('DATAFLOW_TAG', ProspectiveService.DEFAULT_DATAFLOW_TAG)
provenance = RetrospectiveService(dataflow_tag)
input_data = {"bert_base": [["/path/to/bert_base/checkpoint"]], 
              "entailment_config": [["/path/to/entailment/config"]]}

with provenance.get_retrospective_data(ProspectiveService.TF_FINETUNE_BERT, input_data) as result:
    time.sleep(2)
    result['finetuned_bert_model'] = [["1","/path/to/finetuned_bert/checkpoint1"],
                                      ["2","/path/to/finetuned_bert/checkpoint2"]]
