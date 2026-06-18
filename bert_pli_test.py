# -*- coding: utf-8 -*-
__author__ = 'yshao'

import argparse
import os
import json
import logging
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
from tqdm import tqdm

from config_parser import create_config
from reader.reader import init_formatter, init_test_dataset
from model import get_model
from tools.poolout_tool import load_state_keywise
from tools.eval_tool import gen_time_str, output_value
from parse_results import jsonl_to_collie_format, metrics_from_collie_format

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def init_parameters(config, gpu_list, checkpoint, mode, *args, **params):
    result = {}

    logger.info("Begin to initialize dataset and formatter..., mode=%s", mode)
    init_formatter(config, ["test"], *args, **params)
    result["test_dataset"] = init_test_dataset(config, *args, **params)

    logger.info("Begin to initialize models...")
    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    model.set_tokenizer_formatter(mode, config, *args, **params)
    model.set_selection_layer(config.get("model", "selection_mode"))
    
    if len(gpu_list) > 0:
        model = model.cuda()

        try:
            model.init_multi_gpu(gpu_list, config, *args, **params)
        except Exception as e:
            logger.warning("No init_multi_gpu implemented in the model, use single gpu instead.")

    try:
        parameters = torch.load(checkpoint)
        model = load_state_keywise(model, parameters["model"])
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            logger.error(information)
            raise e
        else:
            logger.warning(information)

    result["model"] = model
    
    logger.info("Initialize done.")

    return result

def init_setup(args):
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

    parameters = init_parameters(config, gpu_list, args.checkpoint, "test")

    return config, parameters, gpu_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--checkpoint', help="checkpoint file path", required=True)
    parser.add_argument('--labels-file', help="file path with labels from coliee", required=True)
    parser.add_argument('--result-file', help="result file path", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    
    args = parser.parse_args()

    config, parameters, gpu_list = init_setup(args)

    model = parameters['model']
    dataset = parameters['test_dataset']
    model.eval()

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = "testing"

    output_time = config.getint("output", "output_time")
    step = -1
    result = []

    for step, data in tqdm(enumerate(dataset), desc="Batches", total=len(dataset), ncols=100, leave=False):
        results = model(data, config, gpu_list, acc_result, "test")
        result = result + results["output"]
        cnt += 1
        
        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(0, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)
    
    delta_t = timer() - start_time
    output_info = "testing"
    output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)
    
    raw_results_file = args.result_file.split('.json')[0] + '_raw.json'
    json.dump(result, open(raw_results_file, "w", encoding="utf8"), ensure_ascii=False, sort_keys=True, indent=2)
    
    coliee_result = jsonl_to_collie_format(result)
    with open(args.labels_file, 'r') as f:
        predicted = json.load(f)
    metrics = metrics_from_collie_format(coliee_result, predicted, k_values=[1, 5, 10])
    logger.info("Evaluation Metrics: %s", json.dumps(metrics, indent=2))
    
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {args.result_file}")