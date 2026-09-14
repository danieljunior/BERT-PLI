# Guia de Sumário Quantitativo de Atenção (`attention_summary.py`)

**Data:** 14 de Setembro de 2026  
**Projeto:** BERT-PLI (Legal Case Retrieval - COLIEE)  
**Script Responsável:** [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py)  
**Dependências Dedicadas:** [`requirements_mcnemar.txt`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/requirements_mcnemar.txt)

---

## 1. Visão Geral

O script [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py) realiza a extração e sumarização **quantitativa** dos pesos de atenção em nível de documento/parágrafo para modelos recorrentes baseados em atenção (**AttenRNN**: `GRU` e `LSTM`).

Complementando a análise qualitativa via heatmaps (realizada pelo script de visualização), este script agrega as propriedades matemáticas da distribuição de atenção sobre todo o conjunto de teste, permitindo comparar como a atenção se comporta em pares relevantes (positivos) versus irrelevantes (negativos).

---

## 2. Métricas Calculadas por Par

Para cada amostra $k$ (par de documentos), a distribuição softmax de atenção sobre os $M$ segmentos é denotada por $w = [w_1, w_2, \dots, w_M]$, onde $\sum_{i=1}^M w_i = 1$.

| Métrica | Fórmula / Definição | Interpretação Prática |
| :--- | :--- | :--- |
| **Entropia de Shannon ($H$)** | $H(w) = -\sum_{i=1}^M w_i \ln(w_i + \epsilon)$ | Quantifica o grau de dispersão da atenção. Valore altos indicam atenção difusa/uniforme; valores baixos indicam foco concentrado. |
| **Peso Máximo ($\max w_i$)** | $\max_{1 \le i \le M} w_i$ | Intensidade do segmento de maior relevância no documento candidato. |
| **Índice do Máximo ($\arg\max w_i$)** | $\arg\max_{1 \le i \le M} w_i$ | Posição física (índice 0-indexed) do segmento que recebeu o maior foco. |
| **Peso Mínimo ($\min w_i$)** | $\min_{1 \le i \le M} w_i$ | Nível de atenção residual nos segmentos menos relevantes. |
| **Top-$K$ Concentração** | $\sum_{i \in \text{Top-}K} w_i$ | Proporção acumulada de atenção retida pelos $K$ segmentos mais importantes (padrão: $K=3$). |
| **Índice de Gini ($G$)** | $G = \frac{2 \sum_{i=1}^M i \cdot w_{(i)}}{M \sum_{i=1}^M w_i} - \frac{M+1}{M}$ | Medida de desigualdade da distribuição de atenção em $[0, 1]$. $G \approx 0$ indica distribuição uniforme; $G \approx 1$ indica máxima concentração. |

---

## 3. Seleção Dinâmica do Melhor Checkpoint

O script integra-se com a função `find_best_checkpoint` do módulo [`run_best_model.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/run_best_model.py#L40):
1. Lê o arquivo JSON de métricas de validação especificado em `--metrics` (ex: `output/results/vanilla/v1_attengru_valid_metrics.json`).
2. Identifica automaticamente a época de treino com a maior pontuação de **F1-Score**.
3. Aplica tradução de caminho (via `--path_prefix`), convertendo caminhos contêinerizados (ex: `/app/output/checkpoints/...`) para o caminho local (`./output/checkpoints/...`).

---

## 4. Instruções de Execução

Em um ambiente com PyTorch e suporte de hardware/Docker instalado:

```bash
# Execução básica para modelo Vanilla GRU
python3 attention_summary.py \
    --config config/nlp/divergent/vanilla_gru.config \
    --metrics output/results/vanilla/v1_attengru_valid_metrics.json \
    --ground_truth data/COLIEE/task1_test_labels_2024.json \
    --output output/attention_stats/vanilla_gru \
    --plots \
    --top_k 3
```

---

## 5. Estrutura dos Arquivos de Saída

O diretório informado em `--output` conterá os seguintes arquivos:

```text
output/attention_stats/vanilla_gru/
├── pair_stats.csv           # Tabela contendo uma linha por par avaliado no conjunto de teste
├── aggregate_stats.json     # Estatísticas agregadas (média, desvio padrão, percentis, por label)
└── histograms/              # (Gerado quando a flag --plots é utilizada)
    ├── entropy.png
    ├── max_weight.png
    ├── min_weight.png
    ├── top3_concentration.png
    ├── gini.png
    └── argmax_distribution.png
```

### Exemplo de Estrutura do `aggregate_stats.json`

```json
{
  "n_samples": 3123,
  "entropy": {
    "mean": 3.214,
    "std": 0.382,
    "median": 3.241,
    "p25": 2.980,
    "p75": 3.471,
    "min": 1.021,
    "max": 3.851
  },
  "max_weight": { ... },
  "argmax_weight": {
    "value_counts": {
      "0": 41,
      "1": 53
    }
  },
  "by_label": {
    "positive_count": 1562,
    "negative_count": 1561,
    "positive": {
      "entropy": { "mean": 3.18, "std": 0.35, "median": 3.20 }
    },
    "negative": {
      "entropy": { "mean": 3.24, "std": 0.40, "median": 3.28 }
    }
  }
}
```
