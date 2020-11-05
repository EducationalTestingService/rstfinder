#!/usr/bin/env python
"""
Split RSTDB training set into smaller training + development set.

This script splits up the RST discourse treebank training set JSON file
into  new, smaller training set and a development set.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import json
import random


def main():  # noqa: D103
    """
    Main function.

    Args:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--orig_training_set",
                        default="rst_discourse_tb_edus_TRAINING.json")
    parser.add_argument("--new_training_set",
                        default="rst_discourse_tb_edus_TRAINING_TRAIN.json")
    parser.add_argument("--new_dev_set",
                        default="rst_discourse_tb_edus_TRAINING_DEV.json")
    parser.add_argument("--random_seed", type=int, default=1234567890)
    args = parser.parse_args()

    # load the documents in the original training set and shuffle them
    with open(args.orig_training_set) as f:
        data = json.load(f)
    random.seed(args.random_seed)
    random.shuffle(data)

    # reserve 40 documents for development (the test set has only 38)
    split_point = 40
    train_data = data[split_point:]
    dev_data = data[:split_point]

    # write out the new sets
    with open(args.new_training_set, 'w') as f:
        json.dump(train_data, f)

    with open(args.new_dev_set, 'w') as f:
        json.dump(dev_data, f)


if __name__ == "__main__":
    main()
