# Relatório de Análise Estatística: Teste de McNemar (Parsed Results)

**Data de Atualização:** 14 de Setembro de 2026  
**Projeto:** BERT-PLI (Legal Case Retrieval - COLIEE)  
**Script Responsável:** [`mcnemar_test.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/mcnemar_test.py)  
**Entrada de Predições:** Arquivos `*_parsed_results.json` (`paragraph/`, `summarized/`, `vanilla/`)  
**Ground Truth:** [`data/COLIEE/task1_test_labels_2024.json`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/data/COLIEE/task1_test_labels_2024.json)  
**Resultados Brutos Exportados:** [`output/results/mcnemar_results.csv`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/output/results/mcnemar_results.csv)

---

## 1. Resumo das Ações e Metodologia

1. **Adequação do Formato de Entrada**:
   - Tanto o ground truth (`task1_test_labels_2024.json`) quanto as predições geradas pelos modelos (`*_parsed_results.json`) compartilham o mesmo formato JSON padrão do COLIEE:
     ```json
     {
       "085679.txt": ["018047.txt"],
       "086079.txt": ["058944.txt", "078292.txt"]
     }
     ```
   - Cada relação mapeada `{"query.txt": ["candidato.txt"]}` representa uma predição binária positiva: o par `query_candidato` foi classificado como relevante ($y=1$).
   - Todos os demais pares candidatos não listados para aquela query são classificados como irrelevantes ($y=0$).

2. **Matriz de Contingência 2×2 Pareada**:
   Para quaisquer dois modelos (Modelo A e Modelo B) avaliados sobre o universo de pares da representação:
   - $b_{11}$: Ambos os modelos acertam a classificação do par.
   - $b_{10}$: Modelo A acerta e Modelo B erra.
   - $b_{01}$: Modelo A erra e Modelo B acerta.
   - $b_{00}$: Ambos os modelos erram a classificação do par.

3. **Cálculo Estatístico do Teste de McNemar**:
   - Aplica a estatística de qui-quadrado com correção de continuidade de Yates:
     $$\chi^2 = \frac{(|b_{10} - b_{01}| - 1)^2}{b_{10} + b_{01}}, \quad \text{com } p\text{-valor associado para } \text{gl} = 1$$
   - Caso $b_{10} + b_{01} < 25$, o teste exato binomial é executado automaticamente.
   - Nível de significância: $\alpha = 0.05$.

---

## 2. Panorama Geral das Comparações (108 Pares)

Foram executadas **108 comparações par a par** (36 por representação, cobrindo todas as combinações entre os 9 modelos: `v1_gru`, `v1_lstm`, `v1_transformer`, `v2_gru`, `v2_lstm`, `v2_transformer`, `v3_gru`, `v3_lstm`, `v3_transformer`):

| Grupo / Representação | Universo de Pares | Total de Comparações | Diferenças Significativas ($p < 0.05$) | Sem Diferença Significativa ($p \ge 0.05$) |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla** | 6.062 pares | 36 | **31** (86.1%) | **5** (13.9%) |
| **Summarized** | 5.946 pares | 36 | **27** (75.0%) | **9** (25.0%) |
| **Paragraph** | 5.302 pares | 36 | **22** (61.1%) | **14** (38.9%) |
| **Total Global** | — | **108** | **80** (74.1%) | **28** (25.9%) |

---

## 3. Análise Detalhada por Representação

### 3.1. Vanilla (Texto Integral dos Casos)

O modo `vanilla` apresentou os maiores valores absolutos de F1 e revocação do benchmark nas versões iniciais baseadas em GRU e LSTM.

#### Ranking de Modelos no Grupo Vanilla

| Posição | Modelo | F1-Score | Precisão | Revocação | Acurácia Global |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1º** | **`v1_gru`** | **0.3880** | **29.11%** | **58.19%** | **52.71%** |
| **2º** | **`v1_lstm`** | **0.1439** | **10.79%** | **21.57%** | **33.83%** |
| **3º** | **`v3_gru`** | **0.0990** | **7.43%** | **14.85%** | **30.37%** |
| **4º** | **`v3_lstm`** | **0.0760** | **5.70%** | **11.40%** | **28.59%** |
| **5º** | **`v2_lstm`** | **0.0615** | **4.61%** | **9.22%** | **27.47%** |
| **6º** | **`v3_transformer`** | **0.0615** | **4.61%** | **9.22%** | **27.47%** |
| **7º** | **`v2_gru`** | **0.0555** | **4.16%** | **8.32%** | **27.00%** |
| **8º** | **`v1_transformer`** | **0.0444** | **3.33%** | **6.66%** | **26.15%** |
| **9º** | **`v2_transformer`** | **0.0201** | **1.50%** | **3.01%** | **24.27%** |

#### Destaques Estatísticos (Vanilla)
- **Dominância do `v1_gru`**: O modelo `v1_gru` superou **todos** os demais modelos com a maior significância estatística de todo o estudo:
  - vs. `v3_transformer`: $\chi^2 = 786.09, p = 5.70 \times 10^{-173}$
  - vs. `v2_transformer`: $\chi^2 = 781.24, p = 6.46 \times 10^{-172}$
  - vs. `v3_lstm`: $\chi^2 = 751.59, p = 1.81 \times 10^{-165}$
  - vs. `v1_lstm`: $\chi^2 = 533.67, p = 4.70 \times 10^{-118}$
- **Empates Estatísticos Identificados**:
  - `v2_lstm` vs `v3_transformer`: $\chi^2 = 0.0008, p = 0.9774$ (Desempenho rigorosamente idêntico).
  - `v2_gru` vs `v2_lstm`: $\chi^2 = 0.8699, p = 0.3510$.
  - `v2_gru` vs `v3_transformer`: $\chi^2 = 0.6035, p = 0.4373$.
  - `v1_transformer` vs `v2_gru`: $\chi^2 = 1.8037, p = 0.1793$.
  - `v2_lstm` vs `v3_lstm`: $\chi^2 = 3.6437, p = 0.0563$.

---

### 3.2. Summarized (Resumos dos Casos)

No cenário `summarized`, a rede `v1_lstm` obteve o melhor equilíbrio de precisão e revocação entre todos os modelos avaliados com resumos.

#### Ranking de Modelos no Grupo Summarized

| Posição | Modelo | F1-Score | Precisão | Revocação | Acurácia Global |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1º** | **`v1_lstm`** | **0.2395** | **17.96%** | **35.92%** | **40.08%** |
| **2º** | **`v1_gru`** | **0.1554** | **11.66%** | **23.30%** | **33.45%** |
| **3º** | **`v2_gru`** | **0.0841** | **6.31%** | **12.61%** | **27.83%** |
| **4º** | **`v2_transformer`** | **0.0768** | **5.76%** | **11.52%** | **27.26%** |
| **5º** | **`v3_lstm`** | **0.0717** | **5.38%** | **10.76%** | **26.86%** |
| **6º** | **`v2_lstm`** | **0.0559** | **4.19%** | **8.39%** | **25.61%** |
| **7º** | **`v3_transformer`** | **0.0529** | **3.97%** | **7.94%** | **25.38%** |
| **8º** | **`v1_transformer`** | **0.0495** | **3.71%** | **7.43%** | **25.11%** |
| **9º** | **`v3_gru`** | **0.0495** | **3.71%** | **7.43%** | **25.11%** |

#### Destaques Estatísticos (Summarized)
- **Superioridade do `v1_lstm`**:
  - vs. `v1_transformer`: $\chi^2 = 344.52, p = 6.63 \times 10^{-77}$
  - vs. `v3_transformer`: $\chi^2 = 335.44, p = 6.27 \times 10^{-75}$
  - vs. `v3_gru`: $\chi^2 = 324.43, p = 1.57 \times 10^{-72}$
  - vs. `v1_gru`: $\chi^2 = 80.84, p = 2.45 \times 10^{-19}$
- **Empates Estatísticos**:
  - `v1_transformer` vs `v3_gru`: $\chi^2 = 0.0007, p = 0.9793$.
  - `v2_lstm` vs `v3_transformer`: $\chi^2 = 0.1392, p = 0.7091$.
  - `v3_gru` vs `v3_transformer`: $\chi^2 = 0.2363, p = 0.6269$.
  - `v2_gru` vs `v2_transformer`: $\chi^2 = 1.2403, p = 0.2654$.

---

### 3.3. Paragraph (Parágrafos Individuais)

A granularidade por parágrafos apresentou baixa revocação e F1 inferior em quase todas as variantes, com o `v1_gru` mantendo-se como o único modelo com ganho estatisticamente relevante.

#### Ranking de Modelos no Grupo Paragraph

| Posição | Modelo | F1-Score | Precisão | Revocação | Acurácia Global |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1º** | **`v1_gru`** | **0.0952** | **7.14%** | **14.28%** | **20.05%** |
| **2º** | **`v1_lstm`** | **0.0623** | **4.67%** | **9.35%** | **17.14%** |
| **3º** | **`v2_transformer`** | **0.0120** | **0.90%** | **1.79%** | **12.69%** |
| **4º** | **`v3_transformer`** | **0.0085** | **0.64%** | **1.28%** | **12.39%** |
| **5º** | **`v1_transformer`** | **0.0038** | **0.29%** | **0.58%** | **11.98%** |
| **6º** | **`v2_gru`** | **0.0017** | **0.13%** | **0.26%** | **11.79%** |
| **7º** | **`v3_gru`** | **0.0017** | **0.13%** | **0.26%** | **11.79%** |
| **8º** | **`v2_lstm`** | **0.0009** | **0.06%** | **0.13%** | **11.71%** |
| **9º** | **`v3_lstm`** | **0.0004** | **0.03%** | **0.06%** | **11.67%** |

#### Destaques Estatísticos (Paragraph)
- Os modelos `v2_gru` e `v3_gru` apresentaram predições idênticas ($\chi^2 = 0.0, p = 1.0$).
- 14 comparações (38.9%) não apresentaram diferença estatística significativa, evidenciando o colapso de predições nesta granularidade.

---

## 4. Conclusões Finais

1. **Melhor Modelo Global**:
   - **`vanilla/v1_gru`** atingiu o maior F1-score (**0.3880**) e a maior taxa de revocação (**58.19%**) em todo o benchmark, confirmando superioridade estatística inequívoca ($p < 10^{-100}$) contra todos os competidores.

2. **Melhor Modelo no Cenário de Sumarização**:
   - **`summarized/v1_lstm`** lidera entre os modelos com resumos com F1 de **0.2395** e revocação de **35.92%**, superando significativamente ($p < 10^{-18}$) todos os modelos da mesma família.

3. **Arquiteturas Recorrentes vs. Transformers**:
   - Nas tarefas baseadas nos arquivos `*_parsed_results.json`, as redes recorrentes (**GRU** e **LSTM**) na configuração `v1` superaram os modelos baseados em **Transformer** em precisão e revocação em todos os três cenários avaliados.
