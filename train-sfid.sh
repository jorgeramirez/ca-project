#!/bin/bash

cd SF-ID-Network-For-NLU/

python train.py --dataset=atis --priority_order=slot_first --use_crf=True > /tmp/train_sfid.log 2>&1 &


# using glove embeddings
python train.py --dataset=atis --priority_order=slot_first --use_crf=True --embed_dim=300 --embedding_path=../glove.6B/glove.6B.300d.txt > /tmp/train_sfid.log 2>&1 &
