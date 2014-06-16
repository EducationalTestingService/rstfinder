#!/usr/bin/env python

import argparse
import logging
import re

from nltk.tree import ParentedTree

from discourseparsing.tree_util import TREE_PRINT_MARGIN, _ptb_paren_mapping


def fix_rst_treebank_tree_str(rst_tree_str):
    '''
    This removes some unexplained comments in two files that cannot be parsed.
    - data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_2353.out.dis
    - data/RSTtrees-WSJ-main-1.0/TRAINING/wsj_2367.out.dis
    '''
    return re.sub(r'\)//TT_ERR', ')', rst_tree_str)


def convert_parens_in_rst_tree_str(rst_tree_str):
    '''
    This converts any brackets and parentheses in the EDUs of the RST discourse
    treebank to look like Penn Treebank tokens (e.g., -LRB-),
    so that the NLTK tree API doesn't crash when trying to read in the
    RST trees.
    '''
    for bracket_type, bracket_replacement in _ptb_paren_mapping.items():
        rst_tree_str = re.sub('(_![^_(?=!)]*)\\{}([^_(?=!)]*_!)'.format(bracket_type),
                              '\\g<1>{}\\g<2>'.format(bracket_replacement),
                              rst_tree_str)
    return rst_tree_str


def _delete_span_leaf_nodes(tree):
    subtrees = []
    subtrees.extend([s for s in tree.subtrees()
                     if s != tree and
                     (s.label() == 'span' or s.label() == 'leaf')])

    if len(subtrees) > 0:
        parent = subtrees[0].parent()
        parent.remove(subtrees[0])
        _delete_span_leaf_nodes(tree)


def _move_rel2par(tree):
    subtrees = []
    subtrees.extend(
        [s for s in tree.subtrees() if s != tree and (s.label() == 'rel2par')])

    if subtrees:
        # there should only be one word describing the rel2par
        relation = ' '.join(subtrees[0].leaves())
        parent = subtrees[0].parent()
        # rename the parent node

        parent.set_label('{}:{}'.format(parent.label(), relation).lower())

        # and then delete the rel2par node
        parent.remove(subtrees[0])
        _move_rel2par(tree)


def reformat_rst_tree(input_tree):
    '''
    This method will reformat an RST tree to look a bit more like a Penn
    Treebank tree.

    Note that this modifies the tree in place and does not return a value.
    '''
    logging.debug('Reformatting {}'.format(
        input_tree.pprint(margin=TREE_PRINT_MARGIN)))

    # 1. rename the top node
    input_tree.set_label('ROOT')

    # 2. delete all of the span and leaf nodes (they seem to be just for
    # book keeping)
    _delete_span_leaf_nodes(input_tree)

    # 3. move the rel2par label up to be attached to the Nucleus/Satellite
    # node
    _move_rel2par(input_tree)

    logging.debug('Reformatted: {}'.format(
        input_tree.pprint(margin=TREE_PRINT_MARGIN)))


def main():
    parser = argparse.ArgumentParser(
        description="Converts the gold standard rst parses in the rst treebank to look more like what the parser produces",
        conflict_handler='resolve', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--inputfile',
                        help='Input gold standard rst parse from treebank', type=str, required=True)

    args = parser.parse_args()
    # initialize the loggers
    logging.basicConfig()

    with open(args.inputfile) as f:
        rst_tree_str = f.read().strip()
        rst_tree_str = fix_rst_treebank_tree_str(rst_tree_str)
        rst_tree_str = convert_parens_in_rst_tree_str(rst_tree_str)
        t = ParentedTree(rst_tree_str)
        reformat_rst_tree(t)
        print(t.pprint(margin=TREE_PRINT_MARGIN))


if __name__ == '__main__':
    main()
