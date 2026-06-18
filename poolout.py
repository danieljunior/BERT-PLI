# -*- coding: utf-8 -*-
__author__ = 'yshao'

import argparse
import os
import torch
import json
import logging

from tools.init_tool import init_all
from tools.poolout_tool import pool_out
from config_parser import create_config

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
    parser.add_argument('--result', help="result file path", required=True)
    parser.add_argument('--test', help="test mode", action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.result):
        logger.info(f"Result file already exists: {args.result}")
        exit(0)

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

    dataflow_tag = os.getenv('DATAFLOW_TAG', ProspectiveService.DEFAULT_DATAFLOW_TAG)
    provenance = RetrospectiveService(dataflow_tag)
    poolout_config = None
    with open(configFilePath, 'r', encoding='utf-8') as f:
        poolout_config = f.read()
    if args.test:
        dataset = ProspectiveService.DT_TEST_POOLOUT_CONFIG
        task = ProspectiveService.TF_TEST_POOLOUT
        result_key = ProspectiveService.DT_TEST_POOLOUT_DATA
    else:
        dataset = ProspectiveService.DT_TRAIN_POOLOUT_CONFIG
        task = ProspectiveService.TF_TRAIN_POOLOUT
        result_key = ProspectiveService.DT_TRAIN_POOLOUT_DATA
    input_data = {dataset: [[poolout_config, args.checkpoint]],}
    with provenance.get_retrospective_data(task, input_data) as result:

        if not os.path.exists(args.result):
            out_file = open(args.result, 'w', encoding='utf-8')
            parameters = init_all(config, gpu_list, args.checkpoint, "poolout")
            outputs = pool_out(parameters, config, gpu_list, args.result)
            logger.info(f"Total number of outputs: {outputs}")
            for output in outputs:
                tmp_dict = {
                    'id_': output[0],
                    'res': output[1]
                }
                out_line = json.dumps(tmp_dict, ensure_ascii=False) + '\n'
                out_file.write(out_line)
            out_file.close()
        else:
            logger.info(f"Result file already exists: {args.result}")

        sentences_file = config.get("data", "test_data_path") + "/" + config.get("data", "test_file_list")
        result[result_key] = [["1", args.result, sentences_file]]
    # train(parameters, config, gpu_list)
    logger.info("Poolout completed")