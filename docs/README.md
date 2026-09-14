# Documentação do Projeto BERT-PLI (Análises Estatísticas e Atenção)

Esta pasta reúne a documentação técnica, manuais de uso e relatórios analíticos desenvolvidos para a avaliação dos modelos de recuperação de jurisprudência (Legal Case Retrieval - COLIEE).

---

## 1. Índice de Documentos

- [`mcnemar_analysis_report.md`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/docs/mcnemar_analysis_report.md): **Relatório da Análise Estatística Pareada (McNemar Test)**. Contém os resultados das 108 comparações par a par entre os modelos (`v1`, `v2`, `v3` nas abordagens `paragraph`, `summarized` e `vanilla`) calculadas sobre os arquivos `*_parsed_results.json`.
- [`attention_summary_guide.md`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/docs/attention_summary_guide.md): **Guia de Sumarização Quantitativa de Atenção**. Documentação completa do script [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py), contendo fórmulas matemáticas (entropia, peso máximo, argmax, concentração top-$K$, Gini), parâmetros CLI e especificação dos arquivos de saída.

---

## 2. Scripts Desenvolvidos

### 2.1. Teste de McNemar Pareado (`mcnemar_test.py`)
- **Arquivo**: [`mcnemar_test.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/mcnemar_test.py)
- **Dependências**: [`requirements_mcnemar.txt`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/requirements_mcnemar.txt)
- **Descrição**: Lê os arquivos de predição formatados no padrão COLIEE (`*_parsed_results.json`), alinha as predições com o ground truth (`task1_test_labels_2024.json`) e calcula o teste pareado de McNemar (estatística $\chi^2$ com correção de Yates ou teste exato binomial).
- **Execução**:
  ```bash
  .venv_mcnemar/bin/python mcnemar_test.py
  ```
- **Saída**: Tabela formatada no terminal e arquivo CSV exportado em [`output/results/mcnemar_results.csv`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/output/results/mcnemar_results.csv).

---

### 2.2. Sumário Quantitativo de Atenção (`attention_summary.py`)
- **Arquivo**: [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py)
- **Modelos Suportados**: AttenRNN (`GRU` e `LSTM`)
- **Descrição**: Seleciona dinamicamente o melhor checkpoint a partir das métricas de validação (`run_best_model.py`), processa o conjunto de teste completo e extrai métricas quantitativas dos pesos de atenção ($H$, $\max w_i$, $\arg\max w_i$, Top-$K$, Gini).
- **Execução**:
  ```bash
  python3 attention_summary.py \
      --config config/nlp/divergent/vanilla_gru.config \
      --metrics output/results/vanilla/v1_attengru_valid_metrics.json \
      --ground_truth data/COLIEE/task1_test_labels_2024.json \
      --output output/attention_stats/vanilla_gru \
      --plots
  ```
- **Saída**: `pair_stats.csv`, `aggregate_stats.json` e gráficos PNG em `/histograms`.

---

## 3. Ambientes e Requisitos

- **Ambiente Isolado**: Os pacotes do teste estatístico estão isolados no ambiente `.venv_mcnemar` e salvos em [`requirements_mcnemar.txt`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/requirements_mcnemar.txt).
- **Preservação de Código**: Os arquivos originais [`requirements.txt`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/requirements.txt) e [`extract_attention.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/extract_attention.py) foram estritamente preservados sem alterações.
