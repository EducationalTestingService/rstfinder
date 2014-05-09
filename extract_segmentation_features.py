#!/usr/bin/env python3

'''
A discourse segmenter following the Base model from this paper:
Ngo Xuan Bach, Nguyen Le Minh, Akira Shimazu. 2012.
A Reranking Model for Discourse Segmentation using Subtree Features.
SIGDIAL. http://aclweb.org/anthology//W/W12/W12-1623.pdf.

The output can be fed into CRF++ to train a model
with tune_segmentation_model.py.
'''

import argparse
import json
from nltk.tree import ParentedTree
from discourse_segmentation import extract_segmentation_features
from tree_util import find_first_common_ancestor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='JSON file from convert_rst_discourse_tb.py')
    parser.add_argument('output_path', help='TSV output file to be used by crf++')
    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    with open(args.output_path, 'w') as outfile:
        for doc in data:
            feat_lists, labels = extract_segmentation_features(doc)
            for feat_list, label in zip(feat_lists, labels):
                print('\t'.join(feat_list + [label]), file=outfile)

            print('\t'.join(['' for x in range(len(feat_lists[0]) + 1)]),
                  file=outfile)


if __name__ == '__main__':
    main()
