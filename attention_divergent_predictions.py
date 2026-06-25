import argparse
import os
import torch
from torch.utils.data import DataLoader, DataLoader, Subset
from torch.autograd import Variable
from tqdm import tqdm
from compare_results import load_json
from config_parser.parser import create_config
from extract_attention import (
    plot_attention_heatmap,
    extract_attention_rnn,
    extract_attention_transformer,
)
from tools.init_tool import init_all
from eval_valid import get_best_checkpoint

# MODELS_EVALUATED = ["lstm", "gru", "transformer"]
MODELS_EVALUATED = ["lstm", "gru"]
VARIANTS_EVALUATED = ["vanilla", "summarized"]
# VARIANTS_EVALUATED = ["vanilla", "summarized", "paragraph"]

def load_inter_models_predictions(version, variant):
    """
    Carrega os resultados de predição dos modelos (lstm, gru ) para cada segmentação (vanilla, summarized).
    """

    gru_path = f"output/results/{variant}/{version}_gru_parsed_results.json"
    lstm_path = f"output/results/{variant}/{version}_lstm_parsed_results.json"
    gru = load_json(gru_path)
    lstm = load_json(lstm_path)

    return {"GRU": gru, "LSTM": lstm}

def load_intra_models_predictions(model, version):
    """
    Carrega os resultados de predição das segmentações (vanilla, summarized e paragraph) para cada modelo (lstm, gru, transformer).
    """
    # vanilla_path = f"output/results/{version}/vanilla/{model}_parsed_results.json"
    # summarized_path = f"output/results/{version}/summarized/{model}_parsed_results.json"
    vanilla_path = f"output/results/vanilla/{version}_{model}_parsed_results.json"
    summarized_path = f"output/results/summarized/{version}_{model}_parsed_results.json"
    paragraph_path = None

    vanilla = load_json(vanilla_path)
    summarized = load_json(summarized_path)

    models = {"Vanilla": vanilla, "Summarized": summarized}

    if model != "transformer" and paragraph_path is not None:
        paragraph = load_json(paragraph_path)
        models["Paragraph"] = paragraph

    return models


def calculate_intra_divergent_predictions(version):
    divergent_predictions = {model: {} for model in MODELS_EVALUATED}
    for model in MODELS_EVALUATED:
        models_predictions = load_intra_models_predictions(model, version)
        variants = list(models_predictions.keys())
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                v1, v2 = variants[i], variants[j]
                divergent_predictions[model] = {f"{v1} vs {v2}": []}
                preds1, preds2 = models_predictions[v1], models_predictions[v2]
                if models_predictions[v1] and models_predictions[v2]:
                    common_queries = set(preds1.keys()) & set(preds2.keys())

                    if not common_queries:
                        return 0.0
                    for query in common_queries:
                        set1 = set(preds1[query])
                        set2 = set(preds2[query])

                        symmetric_difference = set1 ^ set2
                        for item in symmetric_difference:
                            divergent_predictions[model][f"{v1} vs {v2}"].append(
                                (query, item)
                            )
    return divergent_predictions

def calculate_inter_divergent_predictions(version):
    divergent_predictions = {variant: {} for variant in VARIANTS_EVALUATED}
    for variant in VARIANTS_EVALUATED:
        models_predictions = load_inter_models_predictions(version, variant)
        variants = list(models_predictions.keys())
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                m1, m2 = variants[i], variants[j]
                divergent_predictions[variant] = {f"{m1} vs {m2}": []}
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
                            divergent_predictions[variant][f"{m1} vs {m2}"].append(
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


def filter_divergent_predictions(dataloader, divergent_predictions):
    dataset = dataloader.dataset
    target_guids = [
        q.split(".txt")[0] + "_" + c.split(".txt")[0] for q, c in divergent_predictions
    ]

    filtered_index = []

    # 2. Varredura para encontrar os índices correspondentes
    # Assumindo que seu dataset retorna um dicionário em cada iteração
    for i in range(len(dataset)):
        item = dataset[i]
        if item["guid"] in target_guids:
            filtered_index.append(i)

    # 3. Crie um objeto Subset com os índices isolados
    subset_dataset = Subset(dataset, filtered_index)

    # 4. Instancie o novo DataLoader preservando as configurações essenciais do original
    return DataLoader(
        subset_dataset,
        batch_size=dataloader.batch_size,
        collate_fn=dataloader.collate_fn,
        num_workers=dataloader.num_workers,
        shuffle=False,  # Raramente faz sentido embaralhar um subset de depuração/filtrado
    )


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--experiment-version",
        "-ev",
        default="v1",
        help="",
    )
    parser.add_argument("--type", default="intra", help="type identifier for the experiment.")
    parser.add_argument("--gpu", default="0", help="gpu id list")
    args = parser.parse_args()

    print("Loading gpu list...")
    use_gpu, gpu_list = init_gpu_list(args.gpu)
 
    if args.type == "intra":
        print("Calculating divergent predictions...")
        divergent_predictions = calculate_intra_divergent_predictions(args.experiment_version)

        for model_name in MODELS_EVALUATED:
            for comparison_models, divergent_values in divergent_predictions[model_name].items():
                variant1, variant2 = comparison_models.split(" vs ")
                for variant in [variant1, variant2]:
                    print("Evaluating model:", model_name, "with variant:", variant) 
                    config = create_config(f"config/nlp/divergent/{variant.lower()}_{model_name.lower()}.config")
                    
                    print("Finding best checkpoint...")
                    checkpoints_path = f"output/checkpoints/{variant.lower()}/{args.experiment_version}_atten{model_name.lower()}"
                    best_checkpoint = get_best_checkpoint(checkpoints_path, config, gpu_list)
                    
                    print("Initializing model and dataset...")
                    parameters = init_all(config, gpu_list, best_checkpoint, "test")
                    model = parameters["model"]
                    dataloader = parameters["test_dataset"]
                    
                    print("Filtering dataloader for divergent predictions...")
                    filtered_dataloader = filter_divergent_predictions(dataloader, divergent_values)
                    for step, data in tqdm(enumerate(filtered_dataloader), total=len(filtered_dataloader)):
                        for key in data.keys():
                            if isinstance(data[key], torch.Tensor):
                                if len(gpu_list) > 0:
                                    data[key] = Variable(data[key].cuda())
                                else:
                                    data[key] = Variable(data[key])
                        results = model(data, config, gpu_list, None, "test")

                        ###### Extração de atenção para visualização ######
                        print("Extracting attention weights for visualization...")
                        if "lstm" in model_name.lower() or "gru" in model_name.lower():
                            attn_w = extract_attention_rnn(model, data, config, gpu_list)
                        else:
                            attn_w = extract_attention_transformer(model, data)

                        print("Plotting attention heatmap...")
                        heatmap_path = (f"./output/results/divergent/{args.experiment_version}/{args.type}{variant.lower()}_{model_name.lower()}/")
                        plot_attention_heatmap(
                            data,
                            attn_w,
                            f"{model_name} - Batch {step}",
                            f"{heatmap_path}{step}.png",
                        )
                        ########################################################
    else:
        #TODO: Implement inter type evaluation
        print("Calculating divergent predictions...")
        divergent_predictions = calculate_inter_divergent_predictions(args.experiment_version)
        for variant in VARIANTS_EVALUATED:
            for comparison_models, divergent_values in divergent_predictions[variant].items():
                model1, model2 = comparison_models.split(" vs ")
                for model_name in [model1, model2]:
                    print("Evaluating model:", model_name, "with variant:", variant) 
                    config = create_config(f"config/nlp/divergent/{variant.lower()}_{model_name.lower()}.config")
                    
                    print("Finding best checkpoint...")
                    checkpoints_path = f"output/checkpoints/{variant.lower()}/{args.experiment_version}_atten{model_name.lower()}"
                    best_checkpoint = get_best_checkpoint(checkpoints_path, config, gpu_list)
                    
                    print("Initializing model and dataset...")
                    parameters = init_all(config, gpu_list, best_checkpoint, "test")
                    model = parameters["model"]
                    dataloader = parameters["test_dataset"]
                    
                    print("Filtering dataloader for divergent predictions...")
                    filtered_dataloader = filter_divergent_predictions(dataloader, divergent_values)
                    for step, data in tqdm(enumerate(filtered_dataloader), total=len(filtered_dataloader)):
                        for key in data.keys():
                            if isinstance(data[key], torch.Tensor):
                                if len(gpu_list) > 0:
                                    data[key] = Variable(data[key].cuda())
                                else:
                                    data[key] = Variable(data[key])
                        results = model(data, config, gpu_list, None, "test")

                        ###### Extração de atenção para visualização ######
                        print("Extracting attention weights for visualization...")
                        if "lstm" in model_name.lower() or "gru" in model_name.lower():
                            attn_w = extract_attention_rnn(model, data, config, gpu_list)
                        else:
                            attn_w = extract_attention_transformer(model, data)

                        print("Plotting attention heatmap...")
                        heatmap_path = (f"./output/results/divergent/{args.experiment_version}/{args.type}{variant.lower()}_{model_name.lower()}/")
                        plot_attention_heatmap(
                            data,
                            attn_w,
                            f"{model_name} - Batch {step}",
                            f"{heatmap_path}{step}.png",
                        )

if __name__ == "__main__":
    main()
