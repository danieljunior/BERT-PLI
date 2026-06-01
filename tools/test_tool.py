import logging
import os
import torch
from timeit import default_timer as timer

from transformers import BertTokenizer
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from tools.eval_tool import gen_time_str, output_value

logger = logging.getLogger(__name__)


# --- [MODIFICAÇÃO 1] Função de Renderização Adicionada ---
def gerar_visualizacao_html(blocos_de_texto, pesos, arquivo_saida):
    if not pesos:
        return
        
    pesos_normalizados = [p / max(pesos) for p in pesos]
    cmap = cm.get_cmap('Reds')
    
    html = '<div style="font-family: Arial; line-height: 1.6; max-width: 800px; padding: 20px;">\n'
    for texto, peso in zip(blocos_de_texto, pesos_normalizados):
        cor_rgba = cmap(peso)
        cor_hex = mcolors.to_hex(cor_rgba)
        # O HTML aplica a cor de fundo com base no peso de atenção
        html += f'<span style="background-color: {cor_hex}; padding: 2px; border-radius: 3px; display: block; margin-bottom: 5px;">{texto}</span>\n'
    html += '</div>'
    
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write(html)
# ---------------------------------------------------------


def test(parameters, config, gpu_list):
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
    
    # --- [MODIFICAÇÃO 2] Inicialização do Tokenizer e Constantes ---
    # Instanciar fora do loop para não sobrecarregar a memória e o tempo de I/O
    tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
    max_para_q = config.getint('model', 'max_para_q')
    max_len = config.getint("data", "max_seq_length")
    
    # Cria diretório para organizar os outputs visuais
    os.makedirs("visualizacoes_atencao", exist_ok=True)
    # -------------------------------------------------------------

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                # --- [MODIFICAÇÃO 3] Correção de Prática Obsoleta ---
                # A classe Variable foi deprecada no PyTorch 0.4. Tensores são nativamente rastreados.
                if len(gpu_list) > 0:
                    data[key] = data[key].cuda()
                else:
                    data[key] = data[key]
                # ----------------------------------------------------

        # A saída agora contém [guid, predição, pesos]
        results = model(data, config, gpu_list, acc_result, "test")
        
        # --- [MODIFICAÇÃO 4] Extração Reversa e Geração Visuais ---
        batch_input_ids = data['input_ids']
        
        for k, example_output in enumerate(results["output"]):
            guid = example_output[0]
            predicao = example_output[1]
            pesos = example_output[2]
            
            # Reconstrói a dimensionalidade para acessar o texto
            # O '-1' engole o max_para_c do seu gerador original
            ids_exemplo = batch_input_ids[k].view(max_para_q, -1, max_len)
            
            blocos_de_texto = []
            for q_idx in range(max_para_q):
                # Extrai os IDs do bloco de query (c=0)
                bloco_ids = ids_exemplo[q_idx, 0, :]
                texto = tokenizer.decode(bloco_ids, skip_special_tokens=True)
                
                if texto.strip():
                    blocos_de_texto.append(texto)
                else:
                    blocos_de_texto.append("[BLOCO VAZIO - PADDING]")
            
            # Gera o Heatmap HTML
            caminho_arquivo = os.path.join("visualizacoes_atencao", f"heatmap_{guid}.html")
            gerar_visualizacao_html(blocos_de_texto, pesos, caminho_arquivo)
            
            # Adiciona APENAS [guid, predição] ao result final 
            # para não quebrar pipelines externos que contam com essa estrutura rígida
            result.append([guid, predicao])
        # ----------------------------------------------------------
        
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