#!/usr/bin/env python3

import argparse
import json
import csv
from nltk.tree import ParentedTree

def extract_features(doc):
    labels = []
    feat_lists = []
    edu_starts = {(x[0], x[1]) for x in doc['edu_start_indices']}
    for sent_num, (sent_tokens, tree_str, sent_tree_positions, pos_tags) in enumerate(zip(doc['tokens'], doc['syntax_trees'], doc['token_tree_positions'], doc['pos_tags'])):
        tree = ParentedTree(tree_str)
        for token_num, (token, tree_position, pos_tag) in enumerate(zip(sent_tokens, sent_tokens, pos_tags)):
            feats = []
            label = 'B-EDU' if (sent_num, token_num) in edu_starts else 'C-EDU'

            feats.append(token.lower())
            feats.append(pos_tag)
            feats.append('B-SENT' if token_num == 0 else 'C-SENT')
            # TODO add parse tree features from http://aclweb.org/anthology//W/W12/W12-1623.pdf

            feat_lists.append(feats)
            labels.append(label)

    return feat_lists, labels



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='JSON file from convert_rst_discourse_tb.py')
    parser.add_argument('output_path', help='TSV output file to be used by crf++')
    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    with open(args.output_path, 'w') as outfile:
        for doc in data:
            feat_lists, labels = extract_features(doc)
            for feat_list, label in zip(feat_lists, labels):
                print('\t'.join(feat_list + [label]), file=outfile)

            print(''.join(['\t' for x in range(len(feat_lists[0]) + 1)]), file=outfile)




if __name__ == '__main__':
    main()
