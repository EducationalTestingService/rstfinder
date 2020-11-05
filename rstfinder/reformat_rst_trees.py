#!/usr/bin/env python

"""
Functions to preprocess the gold standard RST trees.

These functions are needed to fix errors and to make the RST trees
better match the corresponding constituency trees from the PTB.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import logging
import re

from nltk.tree import ParentedTree

from .tree_util import TREE_PRINT_MARGIN, _ptb_paren_mapping


def fix_rst_treebank_tree_str(rst_tree_str):
    """
    Fix errors in some gold standard RST trees.

    This function removes some unexplained comments in two files
    that cannot be parsed.
      - data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_2353.out.dis
      - data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_2367.out.dis
    """
    return re.sub(r'\)//TT_ERR', ')', rst_tree_str)


def convert_parens_in_rst_tree_str(rst_tree_str):
    """
    Convert parentheses in RST trees to match those in PTB trees.

    This function converts any brackets and parentheses in the EDUs of
    the RST discourse treebank to look like Penn Treebank tokens (e.g.,
    -LRB-), so that the NLTK tree API doesn't crash when trying to read
    in the RST trees.
    """
    for bracket_type, bracket_replacement in _ptb_paren_mapping.items():
        rst_tree_str = re.sub(f"(_![^_(?=!)]*)\\{bracket_type}([^_(?=!)]*_!)",
                              f"\\g<1>{bracket_replacement}\\g<2>",
                              rst_tree_str)
    return rst_tree_str


def _delete_span_leaf_nodes(tree):
    """Delete span leaf nodes."""
    subtrees = []
    subtrees.extend([s for s in tree.subtrees()
                     if s != tree and
                     (s.label() == 'span' or s.label() == 'leaf')])

    if len(subtrees) > 0:
        parent = subtrees[0].parent()
        parent.remove(subtrees[0])
        _delete_span_leaf_nodes(tree)


def _move_rel2par(tree):
    """Move the "rel2par" node."""
    subtrees = []
    subtrees.extend(
        [s for s in tree.subtrees() if s != tree and (s.label() == 'rel2par')])

    if subtrees:
        # there should only be one word describing the rel2par
        relation = ' '.join(subtrees[0].leaves())
        parent = subtrees[0].parent()

        # rename the parent node
        parent.set_label(f"{parent.label()}:{relation}".lower())

        # and then delete the rel2par node
        parent.remove(subtrees[0])
        _move_rel2par(tree)


def _replace_edu_strings(input_tree):
    """Replace EDU strings (i.e., the leaves) with indices."""
    edu_index = 0
    for subtree in input_tree.subtrees():
        if isinstance(subtree[0], str):
            subtree.clear()
            subtree.append(edu_index)
            edu_index += 1


def reformat_rst_tree(input_tree):
    """Reformat RST tree to make it look more like a PTB tree."""
    logging.debug(f"Reformatting {input_tree.pformat(margin=TREE_PRINT_MARGIN)}")

    # 1. rename the top node
    input_tree.set_label("ROOT")

    # 2. delete all of the span and leaf nodes (they seem to be just for
    # book keeping)
    _delete_span_leaf_nodes(input_tree)

    # 3. move the rel2par label up to be attached to the Nucleus/Satellite
    # node
    _move_rel2par(input_tree)

    # 4. replace EDU strings with indices
    _replace_edu_strings(input_tree)

    logging.debug(f"Reformatted: {input_tree.pformat(margin=TREE_PRINT_MARGIN)}")


def main():  # noqa: D103
    """
    Main function.

    Args:
    """
    parser = argparse.ArgumentParser(description="Converts the gold standard "
                                                 "RST parses in the RST "
                                                 "treebank to look more like "
                                                 "what the parser produces",
                                     conflict_handler="resolve",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i",
                        "--inputfile",
                        help="Input gold standard RST parse from RST treebank",
                        required=True)
    args = parser.parse_args()

    # initialize the loggers
    logging.basicConfig()

    # process the given input file
    with open(args.inputfile) as inputfh:
        rst_tree_str = inputfh.read().strip()
        rst_tree_str = fix_rst_treebank_tree_str(rst_tree_str)
        rst_tree_str = convert_parens_in_rst_tree_str(rst_tree_str)
        tree = ParentedTree.fromstring(rst_tree_str)
        reformat_rst_tree(tree)
        tree.pprint(margin=TREE_PRINT_MARGIN)


if __name__ == '__main__':
    main()
