#!/bin/bash

cd SlotGated-SLU/

python train.py --dataset=atis > /tmp/train_slot-gated.log 2>&1 &


# using glove embeddings
python train.py --dataset=atis --embed_dim=300 --embedding_path=../glove.6B/glove.6B.300d.txt  > /tmp/train_slot-gated.log 2>&1 &


# start BERT service (use any of the following commands)
bert-serving-start -pooling_strategy NONE -max_seq_len NONE -model_dir ~/uncased_L-12_H-768_A-12/ -device_map 0 > /tmp/bert_service.log 2>&1 &

bert-serving-start -pooling_strategy NONE -max_seq_len NONE -model_dir ~/uncased_L-12_H-768_A-12/ -device_map 0 -pooling_layer -4 -3 -2 -1 > /tmp/bert_service.log 2>&1 &

python start-bert-service.py > /tmp/bert_service.log 2>&1 &


# run training using BERT embeddings
python train.py --dataset=atis --use_bert=True  --embed_dim=768 > /tmp/train_slot-gated.log 2>&1 &

