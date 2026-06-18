import argparse
import json
import logging
import os

import torch
from tensorboardX import SummaryWriter

from config_parser import create_config
from tools.eval_tool import valid
from tools.init_tool import init_all

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments.

    Example:
        args = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="specific config file", required=True)
    parser.add_argument("--gpu", "-g", help="gpu id list")
    parser.add_argument("--checkpoint", help="checkpoint file path")
    parser.add_argument("--checkpoint-dir", help="checkpoint directory path")
    parser.add_argument("--result", help="result file path", required=True)
    return parser.parse_args()


def build_gpu_list(gpu_arg):
    """Build a GPU index list from CLI argument.

    Example:
        gpu_list = build_gpu_list("0,1")
    """
    gpu_list = []
    if gpu_arg is None:
        return gpu_list
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg
    device_list = gpu_arg.split(",")
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))
    return gpu_list


def ensure_single_checkpoint_source(checkpoint_path, checkpoint_dir):
    """Validate checkpoint source arguments.

    Example:
        ensure_single_checkpoint_source("/tmp/1.pkl", None)
    """
    if checkpoint_path and checkpoint_dir:
        raise NotImplementedError(
            "Invalid checkpoint input: checkpoint=%s, checkpoint_dir=%s. Expected exactly one." %
            (checkpoint_path, checkpoint_dir)
        )
    if not checkpoint_path and not checkpoint_dir:
        raise NotImplementedError(
            "Missing checkpoint input: checkpoint=%s, checkpoint_dir=%s. Expected exactly one." %
            (checkpoint_path, checkpoint_dir)
        )


def parse_epoch_from_checkpoint_name(checkpoint_path):
    """Parse epoch from a checkpoint filename like 12.pkl.

    Example:
        epoch = parse_epoch_from_checkpoint_name("/tmp/12.pkl")
    """
    filename = os.path.basename(checkpoint_path)
    if not filename.endswith(".pkl"):
        raise NotImplementedError(
            "Invalid checkpoint filename: %s. Expected numeric '<epoch>.pkl'." % filename
        )
    stem = filename[:-4]
    if not stem.isdigit():
        raise NotImplementedError(
            "Invalid checkpoint filename: %s. Expected numeric '<epoch>.pkl'." % filename
        )
    return int(stem)


def collect_checkpoint_paths(checkpoint_dir):
    """Collect numeric epoch checkpoints from a directory.

    Example:
        checkpoints = collect_checkpoint_paths("/tmp/checkpoints")
    """
    if not os.path.isdir(checkpoint_dir):
        raise NotImplementedError(
            "Invalid checkpoint directory: %s. Expected a directory path." % checkpoint_dir
        )
    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        if not name.endswith(".pkl"):
            continue
        path = os.path.join(checkpoint_dir, name)
        try:
            epoch = parse_epoch_from_checkpoint_name(path)
        except NotImplementedError as exc:
            logger.warning("Skipping checkpoint %s: %s", path, exc)
            continue
        checkpoints.append((epoch, path))
    if not checkpoints:
        raise NotImplementedError(
            "No valid checkpoints in %s. Expected files named '<epoch>.pkl'." % checkpoint_dir
        )
    return sorted(checkpoints, key=lambda item: item[0])


def build_writer(config):
    """Create a SummaryWriter for evaluation-only logs.

    Example:
        writer = build_writer(config)
    """
    tensorboard_root = config.get("output", "tensorboard_path")
    model_name = config.get("output", "model_name")
    log_dir = os.path.join(tensorboard_root, f"{model_name}_eval_only")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir, model_name)


def ensure_cuda_available(gpu_list):
    """Validate CUDA availability for the requested GPUs.

    Example:
        ensure_cuda_available([0])
    """
    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError


def load_checkpoint_state(model, checkpoint_path):
    """Load checkpoint weights into an existing model.

    Example:
        load_checkpoint_state(model, "/tmp/12.pkl")
    """
    try:
        parameters = torch.load(checkpoint_path)
    except Exception as exc:
        raise NotImplementedError(
            "Cannot load checkpoint file %s. Expected a torch checkpoint with key 'model'. Error: %s" %
            (checkpoint_path, exc)
        )
    if "model" not in parameters:
        raise NotImplementedError(
            "Invalid checkpoint data in %s. Expected key 'model'." % checkpoint_path
        )
    try:
        model.load_state_dict(parameters["model"])
    except Exception as exc:
        raise NotImplementedError(
            "Checkpoint state_dict mismatch for %s. Expected compatible model weights. Error: %s" %
            (checkpoint_path, exc)
        )


def evaluate_checkpoint(model, valid_dataset, config, gpu_list, output_function, epoch, writer):
    """Evaluate a model checkpoint on the valid dataset.

    Example:
        metrics = evaluate_checkpoint(model, dataset, config, [], output_fn, 1, writer)
    """
    with torch.no_grad():
        return valid(
            model,
            valid_dataset,
            epoch,
            writer,
            config,
            gpu_list,
            output_function,
        )


def write_result(result_path, checkpoint_dir, results):
    """Write evaluation results to JSON.

    Example:
        write_result("out.json", "/tmp/checkpoints", results)
    """
    payload = {
        "checkpoint_dir": checkpoint_dir,
        "results": results,
    }
    with open(result_path, "w", encoding="utf-8") as out_file:
        json.dump(payload, out_file, ensure_ascii=False, sort_keys=True, indent=2)


def main():
    """Run evaluation for a checkpoint or a directory of checkpoints.

    Example:
        main()
    """
    args = parse_args()
    ensure_single_checkpoint_source(args.checkpoint, args.checkpoint_dir)
    if os.path.exists(args.result):
        print("Output file already exists. Exiting.")
        return

    gpu_list = build_gpu_list(args.gpu)
    os.system("clear")
    config = create_config(args.config)
    ensure_cuda_available(gpu_list)

    if args.checkpoint_dir:
        checkpoints = collect_checkpoint_paths(args.checkpoint_dir)
        checkpoint_dir = args.checkpoint_dir
    else:
        if not os.path.exists(args.checkpoint):
            raise NotImplementedError(
                "Checkpoint does not exist: %s. Expected a valid checkpoint path." % args.checkpoint
            )
        epoch = parse_epoch_from_checkpoint_name(args.checkpoint)
        checkpoints = [(epoch, args.checkpoint)]
        checkpoint_dir = None

    first_epoch, first_path = checkpoints[0]
    parameters = init_all(config, gpu_list, first_path, "train")
    model = parameters["model"]
    valid_dataset = parameters["valid_dataset"]
    output_function = parameters["output_function"]
    writer = build_writer(config)

    results = []
    for epoch, checkpoint_path in checkpoints:
        if checkpoint_path != first_path:
            load_checkpoint_state(model, checkpoint_path)
        metrics = evaluate_checkpoint(
            model,
            valid_dataset,
            config,
            gpu_list,
            output_function,
            epoch,
            writer,
        )
        results.append({
            "checkpoint": checkpoint_path,
            "epoch": epoch,
            "metrics": metrics,
        })

    write_result(args.result, checkpoint_dir, results)
    writer.close()


if __name__ == "__main__":
    main()
