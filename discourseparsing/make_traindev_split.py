#!/usr/bin/env python3

'''
A script to split up the official RST discourse treebank training set into a
new, smaller training set and a development set.
'''

import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--orig_training_set',
                        default='rst_discourse_tb_edus_TRAINING.json')
    parser.add_argument('--new_training_set',
                        default='rst_discourse_tb_edus_TRAINING_TRAIN.json')
    parser.add_argument('--new_dev_set',
                        default='rst_discourse_tb_edus_TRAINING_DEV.json')
    parser.add_argument('--random_seed', type=int, default=1234567890)
    args = parser.parse_args()

    with open(args.orig_training_set) as f:
        data = json.load(f)

    import random
    random.seed(args.random_seed)
    random.shuffle(data)

    # reserve 40 docs for development (the test set has only 38)
    split_point = 40
    train_data = data[split_point:]
    dev_data = data[:split_point]

    with open(args.new_training_set, 'w') as f:
        json.dump(train_data, f)

    with open(args.new_dev_set, 'w') as f:
        json.dump(dev_data, f)

if __name__ == '__main__':
    main()
