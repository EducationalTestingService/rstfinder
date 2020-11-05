#!/usr/bin/env python

"""
Segment discourse units in the given document.

This script segments a document based on a CRF discourse segmentation model
created by ``tune_segmentation_model.py``.

NOTE: This is more an example to be adapted than something that would be
practical since discourse segmentation is an intermediate step that usually
happens inside a discourse parser.
"""

import argparse

from .discourse_segmentation import Segmenter, extract_edus_tokens
from .io_util import read_text_file
from .parse_util import SyntaxParserWrapper
from .tree_util import TREE_PRINT_MARGIN, extract_converted_terminals, extract_preterminals


def main():  # noqa: D103
    """
    Main function.

    Args:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model_path",
                        help="Path to the CRF++ model file.")
    parser.add_argument("input_path",
                        help="Input text document to segment")
    parser.add_argument("-zp",
                        "--zpar_port",
                        required=False,
                        type=int)
    parser.add_argument("-zh",
                        "--zpar_hostname",
                        required=False,
                        default=None)
    parser.add_argument("-zm",
                        "--zpar_model_directory",
                        required=False,
                        default=None)
    args = parser.parse_args()

    raw_text = read_text_file(args.input_path)
    doc_dict = {"doc_id": args.input_path, "raw_text": raw_text}

    parser = SyntaxParserWrapper(port=args.zpar_port,
                                 hostname=args.zpar_hostname,
                                 zpar_model_directory=args.zpar_model_directory)
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
    doc_dict["syntax_trees"] = [t.pformat(margin=TREE_PRINT_MARGIN) for t in trees]
    doc_dict["token_tree_positions"] = token_tree_positions
    doc_dict["pos_tags"] = pos_tags

    segmenter = Segmenter(args.model_path)
    segmenter.segment_document(doc_dict)

    edu_token_lists = extract_edus_tokens(doc_dict["edu_start_indices"], tokens_doc)
    for edu_tokens in edu_token_lists:
        print(' '.join(edu_tokens))


if __name__ == '__main__':
    main()
