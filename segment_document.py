#!/usr/bin/env python3

'''
A script for segmenting a document based on a CRF discourse segmentation model
created by tune_segmentation_model.py.

This is more an example to be adapted than something that would be practical
since discourse segmentation is an intermediate step that should probably
happen within the code for a discourse parser.
'''

import argparse
from tempfile import NamedTemporaryFile
import shlex
import subprocess

from discourse_segmentation import extract_segmentation_features
from tree_util import (extract_preterminals,
                       extract_converted_terminals)
from parse_util import parse_document


def segment_document(doc_dict, model_path):
    # extract features
    # TODO interact with crf++ via cython, etc.?
    tmpfile = NamedTemporaryFile('w')
    feat_lists, _ = extract_segmentation_features(doc_dict)
    for feat_list in feat_lists:
        print('\t'.join(feat_list + ["?"]), file=tmpfile)
    tmpfile.flush()

    # get predictions from the CRF++ model
    crf_output = subprocess.check_output(shlex.split('crf_test -m {} {}'.format(model_path, tmpfile.name))).decode('utf-8').strip()
    tmpfile.close()

    # an index into the list of tokens for this document indicating where the
    # current sentence started
    sent_start_index = 0

    # an index into the list of sentences
    sent_num = 0

    edu_number = 0

    # construct the set of EDU start index tuples (sentence number, token number, EDU number)
    edu_start_indices = []
    all_tokens = doc_dict['tokens']
    cur_sent = all_tokens[0]
    for i, line in enumerate(crf_output.split('\n')):
        if i >= sent_start_index + len(cur_sent):
            sent_start_index += len(cur_sent)
            sent_num += 1
            cur_sent = all_tokens[sent_num] if sent_num < len(all_tokens) else None
        if line.split()[-1] == "B-EDU":
            edu_start_indices.append((sent_num, i - sent_start_index, edu_number))
            edu_number += 1

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
