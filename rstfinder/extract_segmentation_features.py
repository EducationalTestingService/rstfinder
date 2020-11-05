#!/usr/bin/env python

"""
Script to extract features to train discourse segmenter.

This script extracts features for the discourse segmenter (Base model)
described in this paper:

Ngo Xuan Bach, Nguyen Le Minh, Akira Shimazu. 2012.
A Reranking Model for Discourse Segmentation using Subtree Features.
SIGDIAL. http://aclweb.org/anthology//W/W12/W12-1623.pdf.

These features can then be input into CRF++ to train a model
with ``tune_segmentation_model.py``.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import json

from .discourse_segmentation import extract_segmentation_features


def main():  # noqa: D103
    """
    Main function.

    Args:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_path",
                        help="JSON file output from `convert_rst_discourse_tb.py`")
    parser.add_argument("output_path",
                        help="TSV output file to be used by crf++")
    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    with open(args.output_path, 'w') as outfile:
        for doc in data:
            feat_lists_doc, labels_doc = extract_segmentation_features(doc)
            for (feat_lists_sent, labels_sent) in zip(feat_lists_doc, labels_doc):
                for (feat_list, label) in zip(feat_lists_sent, labels_sent):
                    print("\t".join(feat_list + [label]), file=outfile)

                # blank lines between sentences (and documents)
                print(file=outfile)


if __name__ == "__main__":
    main()
