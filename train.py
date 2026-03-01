import argparse
import os
import torch
import logging

from tools.init_tool import init_all
from config_parser import create_config
from tools.train_tool import train

from provenance.retrospective_service import RetrospectiveService
from provenance.prospective_service import ProspectiveService

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    args = parser.parse_args()

    configFilePath = args.config

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    os.system("clear")

    config = create_config(configFilePath)

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    mode = "train"

    dataflow_tag = os.getenv('DATAFLOW_TAG', ProspectiveService.DEFAULT_DATAFLOW_TAG)
    provenance = RetrospectiveService(dataflow_tag)
    config_file = None
    with open(configFilePath, 'r', encoding='utf-8') as f:
        config_file = f.read()
    input_data = {}
    task = None
    if config.get("model", "model_name") == 'BertPoint':
        task = ProspectiveService.TF_FINETUNE_BERT
        bert_path = config.get("model", "bert_path")
        input_data = {ProspectiveService.DT_FINETUNE_CONFIG: [[config_file, bert_path]]}
        output_data = ProspectiveService.DT_FINETUNED_BERT_MODEL
    else:
        task = ProspectiveService.TF_TRAIN_CLASSIFIER
        input_data = {ProspectiveService.DT_CLASSIFIER_CONFIG: [[config_file]]}
        output_data = ProspectiveService.DT_CLASSIFIER_MODEL
    with provenance.get_retrospective_data(task, input_data) as result:

        parameters = init_all(config, gpu_list, args.checkpoint, mode)
        results = train(parameters, config, gpu_list)
        if config.get("model", "model_name") == 'BertPoint':
            results = [item[:2] for item in results]
        # results = [['1', '/app/output/checkpoints/classifier/1.pkl', 'validation_1.json'],
        #             ['2', '/app/output/checkpoints/classifier/2.pkl', 'validation_2.json']]
       
        # finetunebert
        #  results = [['1', '/app/output/checkpoints/bert_finetuned/1.pkl'],
        #                      ['2', '/app/output/checkpoints/bert_finetuned/2.pkl']]
        # train_classifier
        #  results = [['1', '/app/output/checkpoints/classifier/1.pkl', 'validation_1.json'],
#                      ['2', '/app/output/checkpoints/classifier/2.pkl', 'validation_2.json']]
        result[output_data] = results