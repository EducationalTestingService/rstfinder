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
from discourseparsing.io_util import read_text_file


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model_path', help='crf++ model file created by ' +
                        'tune_segmentation_model.py.')
    parser.add_argument('input_path', help='document text file')
    parser.add_argument('-zp', '--zpar_port', type=int)
    parser.add_argument('-zh', '--zpar_hostname', default=None)
    args = parser.parse_args()

    raw_text = read_text_file(args.input_path)
    doc_dict = {"doc_id": args.input_path, "raw_text": raw_text}

    parser = SyntaxParserWrapper(port=args.zpar_port,
                                 hostname=args.zpar_hostname)
    trees, _ = parser.parse_document(doc_dict)
    tokens_doc = [extract_converted_terminals(tree) for tree in trees]
    preterminals = [extract_preterminals(tree) for tree in trees]
    token_tree_positions = [[x.treeposition() for x in
                             preterminals_sentence]
                            for preterminals_sentence
                            in preterminals]
    pos_tags = [[x.label() for x in preterminals_sentence]
                for preterminals_sentence in preterminals]

    doc_dict["tokens"] = tokens_doc
    doc_dict["syntax_trees"] = [t.pprint(TREE_PRINT_MARGIN) for t in trees]
    doc_dict["token_tree_positions"] = token_tree_positions
    doc_dict["pos_tags"] = pos_tags

    segmenter = Segmenter(args.model_path)
    segmenter.segment_document(doc_dict)

    edu_token_lists = extract_edus_tokens(doc_dict['edu_start_indices'],
                                          tokens_doc)
    for edu_tokens in edu_token_lists:
        print(' '.join(edu_tokens))


if __name__ == '__main__':
    main()
