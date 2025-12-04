# -*- coding: utf-8 -*-
__author__ = 'yshao'

import argparse
import os
import json
import logging
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm


from config_parser import create_config

from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from tools.output_init import init_output_function
from tools.poolout_tool import load_state_keywise
from tools.eval_tool import gen_time_str, output_value
from tools.train_tool import checkpoint

from provenance.dfanalyzer_service import DfanalyzerService

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
    model.set_tokenizer_formatter(mode, config, *args, **params)
    model.set_selection_layer(config.get("model", "selection_mode"))

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
    parser.add_argument("--output-path", type=str, default="output/checkpoints/bert-pli", help="output path to save checkpoints")

    args = parser.parse_args()
    dataflow_name = "bert-pli-train"
    provenance_service = DfanalyzerService(dataflow_name)
    provenance_service.create_dataflow()

    config, parameters, gpu_list = init_setup(args)
    
    provenance_service.set_docs_pairs_generation_task(config, "train",)

    model = parameters['model']
    dataset = parameters['train_dataset']
    epoch = config.getint("train", "epoch")
    trained_epoch = parameters["trained_epoch"] + 1
    optimizer = parameters["optimizer"]
    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    output_function = parameters["output_function"]
    output_time = config.getint("output", "output_time")
    global_step = parameters["global_step"]
    total_len = len(dataset)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)
    
    logger.info("Training start....")
    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")
    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num
        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0
        output_info = ""
        step = -1

        for step, data in tqdm(enumerate(dataset), desc="Batches", total=len(dataset), ncols=100, leave=False):
            provenance_service.set_get_example_task(data)
            optimizer.zero_grad()
            
            results = model(data, config, gpu_list, None, "train")
            
            loss, acc_result = results["loss"], results["acc_result"]
            total_loss += float(loss)

            loss.backward()
            optimizer.step()
            
            if step % output_time == 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                            "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)
            
            global_step += 1
            break  # --- REMOVE THIS LINE FOR FULL TRAINING ---

        checkpoint(os.path.join(args.output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config,
                                global_step)