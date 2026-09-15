# Guia de Sumário Quantitativo de Atenção (`attention_summary.py`)

**Data:** 15 de Setembro de 2026  
**Projeto:** BERT-PLI (Legal Case Retrieval - COLIEE)  
**Script Responsável:** [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py)  
**Módulos Auxiliares:** [`attention_metrics.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_metrics.py), [`divergent_subset_loader.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/divergent_subset_loader.py)  
**Dependências Dedicadas:** [`requirements_mcnemar.txt`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/requirements_mcnemar.txt)

---

## 1. Visão Geral

O script [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py) realiza a extração e sumarização **quantitativa** dos pesos de atenção para modelos baseados em RNN (`AttenRNN`: `GRU` e `LSTM`) sobre os **mesmos subconjuntos de dados divergentes** avaliados em [`attention_divergent_predictions.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_divergent_predictions.py).

Enquanto [`attention_divergent_predictions.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_divergent_predictions.py) gera heatmaps qualitativos (PNG) para visualização dos batches, [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py) extrai métricas numéricas agregadas (`pair_stats.csv`, `aggregate_stats.json` e histogramas) sobre exatamente os mesmos pares divergentes.

---

## 2. Tipos de Divergência Suportados

### 2.1. Divergência Intra-Modelo (`--type intra`)
- **Comparações**: Avalia diferenças entre representações de segmentação (`Vanilla vs Summarized`) para cada arquitetura (`GRU` e `LSTM`).
- **Pares Isolados**: Pares presentes na diferença simétrica $P_{\text{vanilla}} \Delta P_{\text{summarized}}$.
- **Diretório de Saída**: `output/results/divergent/{experiment_version}/intra/{variant}_{model}/`

### 2.2. Divergência Inter-Modelo (`--type inter`)
- **Comparações**: Avalia divergências entre modelos (`GRU vs LSTM`) para cada variante de segmentação (`vanilla`, `summarized`, `paragraph`).
- **Pares Isolados**: Pares presentes na diferença simétrica $P_{\text{GRU}} \Delta P_{\text{LSTM}}$.
- **Diretório de Saída**: `output/results/divergent/{experiment_version}/inter/{variant}_{model}/`

---

## 3. Métricas Calculadas por Par

Para cada amostra $k$ (par de documentos), a distribuição softmax de atenção sobre os $M$ segmentos é denotada por $w = [w_1, w_2, \dots, w_M]$, onde $\sum_{i=1}^M w_i = 1$.

| Métrica | Fórmula / Definição | Interpretação Prática |
| :--- | :--- | :--- |
| **Entropia de Shannon ($H$)** | $H(w) = -\sum_{i=1}^M w_i \ln(w_i + \epsilon)$ | Grau de dispersão da atenção. Valores altos indicam atenção difusa; valores baixos indicam foco concentrado. |
| **Peso Máximo ($\max w_i$)** | $\max_{1 \le i \le M} w_i$ | Intensidade do segmento de maior relevância no documento candidato. |
| **Índice do Máximo ($\arg\max w_i$)** | $\arg\max_{1 \le i \le M} w_i$ | Posição física (índice 0-indexed) do segmento que recebeu o maior foco. |
| **Peso Mínimo ($\min w_i$)** | $\min_{1 \le i \le M} w_i$ | Nível de atenção residual nos segmentos menos relevantes. |
| **Top-$K$ Concentração** | $\sum_{i \in \text{Top-}K} w_i$ | Proporção acumulada de atenção retida pelos $K$ segmentos mais importantes (padrão: $K=3$). |
| **Índice de Gini ($G$)** | $G = \frac{2 \sum_{i=1}^M i \cdot w_{(i)}}{M \sum_{i=1}^M w_i} - \frac{M+1}{M}$ | Desigualdade da distribuição em $[0, 1]$. $G \approx 0$ = uniforme; $G \approx 1$ = concentração máxima. |

---

## 4. Instruções de Execução

### 4.1. Divergência Intra-Modelo (Vanilla vs Summarized)
```bash
python3 attention_summary.py \
    --experiment-version v1 \
    --type intra \
    --gpu 0 \
    --plots \
    --top_k 3
```

### 4.2. Divergência Inter-Modelo (GRU vs LSTM)
```bash
python3 attention_summary.py \
    --experiment-version v1 \
    --type inter \
    --gpu 0 \
    --plots \
    --top_k 3
```

---

## 5. Estrutura dos Arquivos de Saída

Cada subpasta gerada em `output/results/divergent/{experiment_version}/{type}/{variant}_{model}/` contém:

```text
output/results/divergent/v1/intra/vanilla_gru/
├── pair_stats.csv           # Tabela contendo uma linha por par divergente avaliado
├── aggregate_stats.json     # Estatísticas agregadas (média, desvio padrão, percentis, por label)
└── histograms/              # (Gerado quando a flag --plots é utilizada)
    ├── entropy.png
    ├── max_weight.png
    ├── min_weight.png
    ├── top3_concentration.png
    ├── gini.png
    └── argmax_distribution.png
```
