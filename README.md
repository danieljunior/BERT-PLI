# BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval


## Build and start docker containers

### DfAnalyzer
- Build: 
  1. Go to `DfAnalyzer-Docker` folder

  2. Build: `docker build --no-cache --tag dfanalyzer .`

  - *On DGX* 
    1. Build on localhost and save as tar file: `docker save custom_dfanalyzer:latest -o custom_dfanalyzer.tar`

    2. Copy to DGX
    
    2. Then load the image: `docker load -i custom_dfanalyzer.tar`

- Start dfanalyzer container:

  `docker run -itd --name dfanalyzer -p 22000:22000 -p 50000:50000 dfanalyzer`
  
  - *On DGX*: 
    - `docker run -itd --name dfanalyzer --shm-size 5gb --security-opt seccomp=unconfined -p 22000:22000 -p 50000:50000 custom_dfanalyzer`

### BERT-PLI

- Build (CUDA 12.4 base + nightly torch/cu124):
`docker build --no-cache --build-arg CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 --tag bert-pli:cuda124-nightly .`

- Start bert-pli container (Instructions to run in nvidia container)
    - `USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} docker run -itd --shm-size 5gb --name bert-pli --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=7 -e DFA_URL=http://dfanalyzer:22000/ -v ${PWD}:/app -v /home/danieljunior/workspace/datasets/jurídicos/COLIEE\ dataset:/app/data --link dfanalyzer:dfanalyzer bert-pli:cuda124-nightly tail -f /dev/null`

    - `NV_GPU=5,7 USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} nvidia-docker run -itd --rm --shm-size 5gb --name bert-pli -v ${PWD}:/app bert-pli:cuda124-nightly tail -f /dev/null`

    - `USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} docker run -itd --rm --shm-size 5gb --name bert-pli --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=7 -v ${PWD}:/app bert-pli:cuda124-nightly tail -f /dev/null`


#### Optional

- When running scripts with `nohup`, use this command to control nohup file size without 
change logrotate setup: `nohup sh -c 'while true; do SIZE=$(stat --printf="%s" nohup.out); if [ $SIZE -gt 10000 ]; then echo "" > nohup.out; fi; sleep 60; done' >/dev/null 2>&1 &  `

- Auxiliary scripts example:

`./run_train.sh config/nlp/BertPoolOutMax.config 0,1,2,3 output/checkpoints/bert_finetuned/1.pkl output/results/vanilla_test_pool_out_max.json test_sumy_sentences.json output/results/vanilla_test_pool_out_max.json output/results/vanilla_test_poolout.json config/nlp/AttenLSTM.config 0 config/nlp/AttenGRU.config 1`

`./run_test_poolout.sh config/nlp/BertPoolOutMax.config 0,1,2,3 output/checkpoints/bert_finetuned/1.pkl output/results/vanilla_test_pool_out_max.json data/test_sumy_sentences.json output/results/vanilla_test_poolout.json config/nlp/BertPoolOutMax_sumy.config output/results/summarized_test_pool_out_max.json data/test_summarized_sentences.json output/results/summarized_test_poolout.json`

`nohup ./run_test.sh config/nlp/AttenGRU.config 0 output/checkpoints/vanilla_attengru/59.pkl output/results/vanilla/gru_results.json output/results/vanilla/gru_parsed_results.json data/old/task1_test_labels_2024.json output/results/vanilla/gru_metrics.json &> test_vanilla_gru.log &`

`nohup ./run_test.sh config/nlp/AttenLSTM.config 1 output/checkpoints/vanilla_attenlstm/59.pkl output/results/vanilla/lstm_results.json output/results/vanilla/lstm_parsed_results.json data/old/task1_test_labels_2024.json output/results/vanilla/lstm_metrics.json &> test_vanilla_lstm.log &`

--------------------------------------------------

`python bert_pli_train.py --config config/nlp/BertPLI.config --checkpoint output/checkpoints/bert_finetuned/1.pkl --gpu 0,1`

`python bert_pli_test.py --config config/nlp/BertPLI.config --checkpoint output/checkpoints/bert-pli/4.pkl --labels-file data/valid_labels.json --result-file output/results/tiny-train.json --gpu 0,1`





## Source

[README](https://github.com/ThuYShao/BERT-PLI-IJCAI2020/blob/master/README.md).

**Run scripts**:

`python3 train.py -c config/nlp/BertPoint.config -g 0`

`python3 poolout.py -c config/nlp/BertPoolOutMax.config -g 0 --checkpoint output/checkpoints/bert_finetuned/1.pkl --result output/results/pool_out_max.json`

`python3 poolout_to_train.py -in data/train_paragraphs_processed_data.json -out output/results/pool_out_max.json --result output/results/train_poolout.json`

`nohup python3 train.py -c config/nlp/AttenLSTM.config -g 1 &> nohup2.out &`

`python3 test.py -c config/nlp/AttenGRU.config -g 0 --checkpoint output/checkpoints/attengru/59.pkl --result output/results/gru_results.json`

`python parse_results.py parse output/results/relevant_gru_results.json output/results/relevant_gru_parsed_result.json`

`python parse_results.py evaluate data/task1_test_labels_2024.json output/results/gru_parsed_result.json output/results/gru_metrics.json`

**Citation**:

```
@inproceedings{shao2020bert,
  title={BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval},
  author={Shao, Yunqiu and Mao, Jiaxin and Liu, Yiqun and Ma, Weizhi and Satoh, Ken and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20},
  pages={3501--3507},
  year={2020}
}
```
