# Documentação do Projeto BERT-PLI (Análises Estatísticas e Atenção)

Esta pasta reúne a documentação técnica, manuais de uso e relatórios analíticos desenvolvidos para a avaliação dos modelos de recuperação de jurisprudência (Legal Case Retrieval - COLIEE).

---

## 1. Índice de Documentos

- [`mcnemar_analysis_report.md`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/docs/mcnemar_analysis_report.md): **Relatório da Análise Estatística Pareada (McNemar Test)**. Contém os resultados das 108 comparações par a par entre os modelos (`v1`, `v2`, `v3` nas abordagens `paragraph`, `summarized` e `vanilla`) calculadas sobre os arquivos `*_parsed_results.json`.
- [`attention_summary_guide.md`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/docs/attention_summary_guide.md): **Guia de Sumarização Quantitativa de Atenção para Predições Divergentes**. Documentação do script [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py) e módulos [`attention_metrics.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_metrics.py) / [`divergent_subset_loader.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/divergent_subset_loader.py), detalhando o processamento sobre os mesmos dados de [`attention_divergent_predictions.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_divergent_predictions.py) com os parâmetros `-ev` e `--type`.

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

### 2.2. Sumário Quantitativo de Atenção Divergente (`attention_summary.py`)
- **Arquivo Principal**: [`attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_summary.py)
- **Módulos Auxiliares**: [`attention_metrics.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/attention_metrics.py), [`divergent_subset_loader.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/divergent_subset_loader.py)
- **Modelos Suportados**: AttenRNN (`GRU` e `LSTM`)
- **Parâmetros**:
  - `--experiment-version` / `-ev`: Versão dos experimentos (`v1`, `v2`, `v3`).
  - `--type` / `-t`: Tipo de divergência (`intra` para variantes de segmentação ou `inter` para arquiteturas).
- **Execução**:
  ```bash
  # Modo intra-modelo
  python3 attention_summary.py --experiment-version v1 --type intra --plots

  # Modo inter-modelo
  python3 attention_summary.py --experiment-version v1 --type inter --plots
  ```
- **Saída**: Diretórios estruturados em `output/results/divergent/{experiment_version}/{type}/{variant}_{model}/` contendo `pair_stats.csv`, `aggregate_stats.json` e `/histograms`.

---

## 3. Ambientes e Requisitos

- **Ambiente Isolado**: Os pacotes do teste estatístico estão isolados no ambiente `.venv_mcnemar` e salvos em [`requirements_mcnemar.txt`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/requirements_mcnemar.txt).
- **Testes Unitários**: Testes automatizados cobrindo métricas numéricas e carregadores divergentes em [`tests/test_attention_summary.py`](file:///home/danieljunior/workspace/BERT-PLI-IJCAI2020/tests/test_attention_summary.py).
