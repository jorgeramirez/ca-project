#!/bin/bash

cd SF-ID-Network-For-NLU/
python train.py --dataset=atis --priority_order=slot_first --use_crf=True
