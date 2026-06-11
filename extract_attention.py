import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

# Funções utilitárias de plotagem
def plot_attention_heatmap(attn_weights, title, output_path, max_para_q=None, max_para_c=None):
    """
    Plota a matriz de atenção como um heatmap e salva na pasta.
    Se a dimensão dos pesos for [M], ele tenta formatar para 2D (Q x C) 
    se os valores max_para_q e max_para_c forem fornecidos e corretos.
    """
    plt.figure(figsize=(12, 10))
    
    # attn_weights deve ser [Q, C] idealmente. 
    # Se for 1D, tentaremos formatar para Q x C se bater o tamanho.
    if len(attn_weights.shape) == 1:
        total_len = len(attn_weights)
        if max_para_q and max_para_c and (max_para_q * max_para_c) == total_len:
            attn_matrix = attn_weights.reshape(max_para_q, max_para_c)
        else:
            # Caso os parâmetros não batam, plota como barra ou expande
            attn_matrix = np.expand_dims(attn_weights, axis=0)
    else:
        attn_matrix = attn_weights

    ax = sns.heatmap(attn_matrix, cmap='viridis', 
                xticklabels=[f"C{i}" for i in range(attn_matrix.shape[1])] if len(attn_matrix.shape) == 2 else False,
                yticklabels=[f"Q{i}" for i in range(attn_matrix.shape[0])] if len(attn_matrix.shape) == 2 and attn_matrix.shape[0] > 1 else False)
    
    plt.title(title)
    if len(attn_matrix.shape) == 2:
        plt.xlabel("Case Segments (C)")
        plt.ylabel("Query Segments (Q)")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Heatmap salvo em {output_path}")

# ==========================================
# 1. Extração para AttenRNN (LSTM/GRU)
# ==========================================
def extract_attention_rnn(model, data, config, gpu_list):
    """
    Recalcula a atenção do AttenRNN externamente, sem modificar a classe.
    Utiliza as mesmas operações matemáticas presentes em AttenRNN.Attention
    """
    model.eval()
    with torch.no_grad():
        x = data['input'] # [B, M, I]
        batch_size = x.size(0)
        
        # O model possui um init_hidden
        model.init_hidden(config, batch_size, gpu_list)
        
        # Passa pelo RNN
        rnn_out, _ = model.rnn(x, model.hidden) # [B, M, 2H]
        tmp_rnn = rnn_out.permute(0, 2, 1)      # [B, 2H, M]
        
        # Max pool -> FC -> unsqueeze
        feature = model.max_pool(tmp_rnn).squeeze(2) # [B, 2H]
        feature = model.fc_a(feature).unsqueeze(2)   # [B, 2H, 1]
        
        # Agora recalculamos a atenção (mesmo de model.attention.forward)
        ratio = torch.bmm(rnn_out, feature)          # [B, M, 1]
        ratio = ratio.view(ratio.size(0), ratio.size(1)) # [B, M]
        attn_weights = F.softmax(ratio, dim=1)       # [B, M]
        
        return attn_weights.cpu().numpy()

# ==========================================
# 2. Extração para AttenTransformer
# ==========================================
def extract_attention_transformer(model, data):
    """
    Usa hooks para capturar a saída da layer interna nn.MultiheadAttention 
    dentro do PyTorch's nn.TransformerEncoderLayer do AttenTransformer.
    """
    model.eval()
    attention_weights_list = []
    
    # Define o hook para extrair weights
    def hook_fn(module, input, output):
        # Em nn.MultiheadAttention, output = (attn_output, attn_weights)
        attention_weights_list.append(output[1].detach().cpu().numpy())

    # Registra o hook na primeira layer de transformer (ou em todas)
    # Assumimos que a arquitetura padrão usa model.transformer (que é TransformerEncoder)
    # e dentro possui layers
    hooks = []
    for layer in model.transformer.layers:
        h = layer.self_attn.register_forward_hook(hook_fn)
        hooks.append(h)
    
    # Realiza forward apenas até onde importa (precisamos passar x pelo transformer)
    with torch.no_grad():
        x = data['input']
        _ = model.transformer(x)
        
    # Limpa os hooks para não afetar próximas execuções
    for h in hooks:
        h.remove()
        
    # attention_weights_list agora contém a atenção de todas as camadas
    # Vamos retornar a média de todas as camadas, ou simplesmente a primeira.
    # O shape padrão geralmente é [batch_size, num_heads, seq_len, seq_len] ou [batch_size, seq_len, seq_len] (depende da flag average_attn_weights no PyTorch)
    # Se a flag não estiver ativada/desativada (PyTorch default é True => [B, L, L])
    
    return np.array(attention_weights_list)


if __name__ == "__main__":
    print("Este é um módulo de utilidades. Você pode importar `extract_attention_rnn` e `extract_attention_transformer`.")
    print("Exemplo de uso na sua inferência:")
    print('''
    from extract_attention import extract_attention_rnn, plot_attention_heatmap
    # No loop de inferência/teste:
    attn_w = extract_attention_rnn(model, data, config, gpu_list)
    # O batch possui formato [B, M]. Para o primeiro item:
    plot_attention_heatmap(attn_w[0], "AttenRNN - Exemplo 0", "output/heatmaps/rnn_0.png", max_para_q=10, max_para_c=15)
    ''')
