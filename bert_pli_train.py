# -*- coding: utf-8 -*-
__author__ = 'yshao'

import argparse
import os
import json
import logging

import torch
from torch.autograd import Variable
from tqdm import tqdm

from tools.init_tool import init_all
from tools.poolout_tool import pool_out
from config_parser import create_config
from model.nlp.BertPLI import BertPLI

from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from tools.output_init import init_output_function
from tools.poolout_tool import load_state_keywise

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def init_parameters(config, gpu_list, checkpoint, mode, *args, **params):
    result = {}

    logger.info("Begin to initialize dataset and formatter..., mode=%s", mode)
    if mode == "train":
        init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)
    else:
        init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)

    logger.info("Begin to initialize models...")

    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = 0
    global_step = 0

    if len(gpu_list) > 0:
        model = model.cuda()

        try:
            model.init_multi_gpu(gpu_list, config, *args, **params)
        except Exception as e:
            logger.warning("No init_multi_gpu implemented in the model, use single gpu instead.")

    try:
        parameters = torch.load(checkpoint)
        model.poolout_max = load_state_keywise(model.poolout_max, parameters["model"])

        if mode == "train":
            trained_epoch = parameters["trained_epoch"]
            if config.get("train", "optimizer") == parameters["optimizer_name"]:
                optimizer.load_state_dict(parameters["optimizer"])
            else:
                logger.warning("Optimizer changed, do not load parameters of optimizer.")

            if "global_step" in parameters:
                global_step = parameters["global_step"]
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            logger.error(information)
            raise e
        else:
            logger.warning(information)

    result["model"] = model
    if mode == "train":
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

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

    parameters = init_parameters(config, gpu_list, args.checkpoint, "train")
    # parameters = init_all(config, gpu_list, args.checkpoint, "poolout")

    return config, parameters, gpu_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    args = parser.parse_args()

    config, parameters, gpu_list = init_setup(args)
    model = parameters['model']
    dataset = parameters['train_dataset']
    result = []
    for step, data in tqdm(enumerate(dataset), desc="Training", total=len(dataset), ncols=100, leave=False):

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, None, "train")
        result = result + results["output"]
        # logger.info(f"Result: {results}")
    # outputs = pool_out(parameters, config, gpu_list, args.result)
    # logger.info(f"Total number of outputs: {outputs}")
    # for output in outputs:
    #     tmp_dict = {
    #         'id_': output[0],
    #         'res': output[1]
    #     }
    #     out_line = json.dumps(tmp_dict, ensure_ascii=False) + '\n'
    #     out_file.write(out_line)
    # out_file.close()

    # train(parameters, config, gpu_list)
