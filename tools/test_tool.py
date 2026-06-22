import logging
import os
import torch
from torch.autograd import Variable
from timeit import default_timer as timer

from tools.eval_tool import gen_time_str, output_value
from extract_attention import extract_attention_rnn, extract_attention_transformer, plot_attention_heatmap

logger = logging.getLogger(__name__)


def test(parameters, config, gpu_list, heatmap=False):
    model = parameters["model"]
    dataset = parameters["test_dataset"]
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

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, acc_result, "test")

        ###### Extração de atenção para visualização ######
        if heatmap:
            model_name = config.get("output", "model_name")
            if "lstm" in model_name.lower() or "gru" in model_name.lower():
                attn_w = extract_attention_rnn(model, data, config, gpu_list)
            else:
                attn_w = extract_attention_transformer(model, data)
            heatmap_path = config.get("data", "test_data_path") + f"/heatmaps/{model_name.lower()}/"
            plot_attention_heatmap(data, attn_w, f"{model_name} - Batch {step}", f"{heatmap_path}{step}.png")
        ########################################################

        result = result + results["output"]
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    output_info = "testing"
    output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    return result
