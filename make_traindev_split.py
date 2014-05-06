#!/usr/bin/env python3

import json
with open('rst_discourse_tb_edus_TRAINING.json') as f:
    data = json.load(f)

import random
random.seed(1234567890)
random.shuffle(data)

split_point = round(float(len(data)) / 3.0)
train_data = data[split_point:]
dev_data = data[:split_point]

with open('rst_discourse_tb_edus_TRAINING_TRAIN.json', 'w') as f:
    json.dump(train_data, f)

with open('rst_discourse_tb_edus_TRAINING_DEV.json', 'w') as f:
    json.dump(dev_data, f)
