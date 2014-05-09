#!/usr/bin/env python3

import argparse
import json

from nltk.tree import ParentedTree

from discourse_segmentation import extract_segmentation_features
from tree_util import HeadedParentedTree
from tree_util import (find_first_common_ancestor, extract_preterminals,
                       extract_converted_terminals)
from parse_util import parse_document


def segment_document(doc_dict, model_path):
    # TODO extract features
    # TODO call crf_test (via a subprocess for now) to predict EDU start tokens
    # TODO process crf_test output into a list of tuples

    edu_start_indices = [(x, 0, 0) for x in range(len(doc_dict['tokens']))]  # TODO replace this placeholder line

    # check that all sentences are covered by the output list of EDUs
    assert set(range(len(doc_dict['tokens']))) == {x[0] for x
                                                   in edu_start_indices}

    doc_dict['edu_start_indices'] = edu_start_indices


def extract_edus_tokens(edu_start_indices, tokens_doc):
    res = []

    # add a dummy index pair representing the end of the document
    tmp_indices = edu_start_indices + [[edu_start_indices[-1][0] + 1,
                                        0,
                                        edu_start_indices[-1][2] + 1]]

    for (prev_sent_index, prev_tok_index, prev_edu_index), \
            (sent_index, tok_index, edu_index) \
            in zip(tmp_indices, tmp_indices[1:]):
        if sent_index == prev_sent_index and tok_index > prev_tok_index:
            res.append(tokens_doc[prev_sent_index][prev_tok_index:tok_index])
        elif sent_index > prev_sent_index and tok_index == 0:
            res.append(tokens_doc[prev_sent_index][prev_tok_index:])
        else:
            raise ValueError('An EDU crosses sentences: ({}, {}) => ({}, {})'
                             .format(prev_sent_index, prev_tok_index,
                                     sent_index, tok_index))
    return res


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='crf++ model file created by ' +
                                           'tune_segmentation_model.py.')
    parser.add_argument('input_path', help='document text file')
    args = parser.parse_args()

    with open(args.input_path) as f:
        doc = f.read()

    trees = parse_document(doc)
    tokens_doc = [extract_converted_terminals(tree) for tree in trees]
    preterminals = [extract_preterminals(tree) for tree in trees]
    token_tree_positions = [[x.treeposition() for x in
                             preterminals_sentence]
                            for preterminals_sentence
                            in preterminals]
    pos_tags = [[x.label() for x in preterminals_sentence]
                for preterminals_sentence in preterminals]

    doc_dict = {"tokens": tokens_doc,
                "syntax_trees": [t.pprint() for t in trees],
                "token_tree_positions": token_tree_positions,
                "pos_tags": pos_tags}

    segment_document(doc_dict, args.model_path)

    edu_token_lists = extract_edus_tokens(doc_dict['edu_start_indices'],
                                          tokens_doc)
    for edu_tokens in edu_token_lists:
        print(' '.join(edu_tokens))


if __name__ == '__main__':
    main()
