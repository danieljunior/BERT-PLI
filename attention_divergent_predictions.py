import argparse
import os
import torch
from torch.autograd import Variable
from compare_results import load_json
from config_parser.parser import create_config
from extract_attention import plot_attention_heatmap, extract_attention_rnn, extract_attention_transformer
from tools.init_tool import init_all


def load_models_predictions(model, version):
    """
    Carrega os resultados de predição dos modelos (vanilla, summarized e paragraph) para cada modelo (lstm, gru, transformer).
    """
    vanilla_path = f"output/results/{version}/vanilla/{model}_parsed_results.json"
    summarized_path = f"output/results/{version}/summarized/{model}_parsed_results.json"
    paragraph_path = None

    vanilla = load_json(vanilla_path)
    summarized = load_json(summarized_path)

    models = {"Vanilla": vanilla, "Summarized": summarized}

    if model != "transformer" and paragraph_path is not None:
        paragraph = load_json(paragraph_path)
        models["Paragraph"] = paragraph

    return models

def calculate_divergent_predictions(args):
    models = ["lstm", "gru", "transformer"]
    divergent_predictions = {model: {} for model in models}
    for model in models:
        models_predictions = load_models_predictions(model, args.version)
        model_names = list(models_predictions.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                divergent_predictions[model] = {f"{m1} vs {m2}": []}
                preds1, preds2 = models_predictions[m1], models_predictions[m2]
                if models_predictions[m1] and models_predictions[m2]:
                    common_queries = set(preds1.keys()) & set(preds2.keys())

                    if not common_queries:
                        return 0.0
                    for query in common_queries:
                        set1 = set(preds1[query])
                        set2 = set(preds2[query])

                        symmetric_difference = set1 ^ set2
                        for item in symmetric_difference:
                            divergent_predictions[model][f"{m1} vs {m2}"].append(
                                (query, item)
                            )
    return divergent_predictions

def init_gpu_list(gpu_arg):
    use_gpu = True
    gpu_list = []
    if gpu_arg is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg

        device_list = gpu_arg.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))
    return use_gpu, gpu_list

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--version",
        "-v",
        default="v1",
        help="",
    )
    parser.add_argument("--config", "-c", help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', default="0", help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path", required=True)
    args = parser.parse_args()
    print("Loading configuration...")
    configFilePath = args.config
    config = create_config(configFilePath)
    print("Calculating divergent predictions...")
    divergent_predictions = calculate_divergent_predictions(args)
    print("Loading gpu list..." )
    use_gpu, gpu_list = init_gpu_list(args.gpu)
    print("Initializing model and dataset...")
    parameters = init_all(config, gpu_list, args.checkpoint, "test")
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    for step, data in enumerate(dataset):
        import pdb; pdb.set_trace()
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, None, "test")

        ###### Extração de atenção para visualização ######
        model_name = config.get("output", "model_name")
        if "lstm" in model_name.lower() or "gru" in model_name.lower():
            attn_w = extract_attention_rnn(model, data, config, gpu_list)
        else:
            attn_w = extract_attention_transformer(model, data)
        heatmap_path = config.get("data", "test_data_path") + f"/heatmaps/{model_name.lower()}/"
        plot_attention_heatmap(data, attn_w, f"{model_name} - Batch {step}", f"{heatmap_path}{step}.png")
        ########################################################

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
