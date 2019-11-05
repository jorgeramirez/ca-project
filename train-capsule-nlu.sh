#!/bin/bash

cd Capsule-NLU/

python train.py --dataset=atis  > /tmp/train_capsule_nlu.log 2>&1 &


# using glove embeddings
python train.py --dataset=atis  --embed_dim=300 --embedding_path=../glove.6B/glove.6B.300d.txt > /tmp/train_capsule_nlu.log 2>&1 &


wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip

# start BERT service (use any of the following commands)
bert-serving-start -pooling_strategy NONE -max_seq_len NONE -model_dir ~/uncased_L-12_H-768_A-12/ -device_map 0 > /tmp/bert_service.log 2>&1 &
python start-bert-service.py > /tmp/bert_service.log 2>&1 &


# run training using BERT embeddings
python train.py --dataset=atis  --use_bert=True  --embed_dim=768 --intent_dim=96 > /tmp/train_capsule_nlu.log 2>&1 &


# running tensorboard
tensorboard --logdir model -host 10.128.0.6
