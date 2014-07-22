#!/usr/bin/env python3

'''
A script for segmenting a document based on a CRF discourse segmentation model
created by tune_segmentation_model.py.

This is more an example to be adapted than something that would be practical
since discourse segmentation is an intermediate step that should probably
happen within the code for a discourse parser.
'''

import argparse

from discourseparsing.discourse_segmentation import (Segmenter,
                                                     extract_edus_tokens)
from discourseparsing.tree_util import (extract_preterminals,
                                        extract_converted_terminals,
                                        TREE_PRINT_MARGIN)
from discourseparsing.parse_util import SyntaxParserWrapper


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='crf++ model file created by ' +
                        'tune_segmentation_model.py.')
    parser.add_argument('input_path', help='document text file')
    args = parser.parse_args()

    with open(args.input_path) as f:
        doc = f.read()

    parser = SyntaxParserWrapper()
    trees = parser.parse_document(doc)
    tokens_doc = [extract_converted_terminals(tree) for tree in trees]
    preterminals = [extract_preterminals(tree) for tree in trees]
    token_tree_positions = [[x.treeposition() for x in
                             preterminals_sentence]
                            for preterminals_sentence
                            in preterminals]
    pos_tags = [[x.label() for x in preterminals_sentence]
                for preterminals_sentence in preterminals]

    doc_dict = {"tokens": tokens_doc,
                "syntax_trees": [t.pprint(TREE_PRINT_MARGIN) for t in trees],
                "token_tree_positions": token_tree_positions,
                "pos_tags": pos_tags}

    segmenter = Segmenter(args.model_path)
    segmenter.segment_document(doc_dict)

    edu_token_lists = extract_edus_tokens(doc_dict['edu_start_indices'],
                                          tokens_doc)
    for edu_tokens in edu_token_lists:
        print(' '.join(edu_tokens))


if __name__ == '__main__':
    main()
