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

    dataflow_tag = os.getenv('DATAFLOW_TAG', ProspectiveService.DEFAULT_DATAFLOW_TAG)
    provenance = RetrospectiveService(dataflow_tag)
    poolout_config = None
    with open(configFilePath, 'r', encoding='utf-8') as f:
        poolout_config = f.read()
    input_data = {ProspectiveService.DT_POOLOUT_CONFIG: [[poolout_config, args.checkpoint]],}
    with provenance.get_retrospective_data(ProspectiveService.TF_POOLOUT, input_data) as result:

        parameters = init_all(config, gpu_list, args.checkpoint, "poolout")

        out_file = open(args.result, 'w', encoding='utf-8')
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
        
        sentences_file = config.get("data", "test_data_path") + "/" + config.get("data", "test_file_list")
        result[ProspectiveService.DT_POOLOUT_DATA] = [["1", args.result, sentences_file]]
    # train(parameters, config, gpu_list)
