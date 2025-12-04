# BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval


## Instructions to run in nvidia container

- Start dfanalyzer container

`docker run -it --name dfanalyzer -p 22000:22000 -p 50000:50000 dfanalyzer`

- Start bert-pli container
    - `USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} docker run -itd --shm-size 5gb --name bert-pli --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=7 -e DFA_URL=http://dfanalyzer:22000/ -v ${PWD}:/app -v /home/danieljunior/workspace/datasets/jurídicos/COLIEE\ dataset:/app/data --link dfanalyzer:dfanalyzer bert-pli:latest tail -f /dev/null`

    - `NV_GPU=5,7 USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} nvidia-docker run -itd --rm --shm-size 5gb --name bert-pli -v ${PWD}:/app bert-pli:latest tail -f /dev/null`

    - `USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} docker run -itd --rm --shm-size 5gb --name bert-pli --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=7 -v ${PWD}:/app bert-pli:latest tail -f /dev/null`



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