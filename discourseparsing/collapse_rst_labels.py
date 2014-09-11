#!/usr/bin/env python3

'''
Script to collapse RST discourse treebank relation types,
based on a perl script by Kenji Sagae.
'''

import argparse
import re

from nltk.tree import ParentedTree

from discourseparsing.tree_util import TREE_PRINT_MARGIN
from discourseparsing.reformat_rst_trees import reformat_rst_tree


def collapse_rst_labels(tree):
    '''
    Collapse the RST labels to the 18 described by the Carlson et al. paper
    that comes with the RST discourse treebank.

    This modifies the tree in place.
    '''
    for subtree in tree.subtrees():
        # Walk the tree, modify any terminals whose parent is rel2par.
        subtree.set_label(_collapse_rst_label(subtree.label()))


def _collapse_rst_label(label):
    if not re.search(':', label):
        return label

    direction, relation = label.split(':')

    relation_lc = relation.lower()
    if re.search(r'^attribution', relation_lc):
        relation = "ATTRIBUTION"

    elif re.search(r'^(background|circumstance)', relation_lc):
        relation = "BACKGROUND"

    elif re.search(r'^(cause|result|consequence)', relation_lc):
        relation = "CAUSE"

    elif re.search(r'^(comparison|preference|analogy|proportion)',
                   relation_lc):
        relation = "COMPARISON"

    elif re.search(r'^(condition|hypothetical|contingency|otherwise)',
                   relation_lc):
        relation = "CONDITION"

    elif re.search(r'^(contrast|concession|antithesis)', relation_lc):
        relation = "CONTRAST"

    elif re.search(r'^(elaboration.*|example|definition)', relation_lc):
        relation = "ELABORATION"

    elif re.search(r'^(purpose|enablement)', relation_lc):
        relation = "ENABLEMENT"

    elif re.search(r'^(problem\-solution|question\-answer|statement\-response|topic\-comment|comment\-topic|rhetorical\-question)', relation_lc):
        relation = "TOPICCOMMENT"

    elif re.search(r'^(evaluation|interpretation|conclusion|comment)',
                   relation_lc):
        # note that this check for "comment" needs to come after the one
        # above that looks for "comment-topic"
        relation = "EVALUATION"

    elif re.search(r'^(evidence|explanation.*|reason)', relation_lc):
        relation = "EXPLANATION"

    elif re.search(r'^(list|disjunction)', relation_lc):
        relation = "JOINT"

    elif re.search(r'^(manner|means)', relation_lc):
        relation = "MANNERMEANS"

    elif re.search(r'^(summary|restatement)', relation_lc):
        relation = "SUMMARY"

    elif re.search(r'^(temporal\-.*|sequence|inverted\-sequence)',
                   relation_lc):
        relation = "TEMPORAL"

    elif re.search(r'^(topic-.*)', relation_lc):
        relation = "TOPICCHANGE"

    elif re.search(r'^(span|same\-unit|textualorganization)$', relation_lc):
        pass

    else:
        raise ValueError('unknown relation type in label: {}'.format(label))

    # TODO make this all upper case (to resemble PTB nonterminals)?
    res = "{}:{}".format(direction, relation).lower()

    return res


def main():
    parser = argparse.ArgumentParser(
        description="Note that this main method is just for testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'input_path', help='path to an RST discourse treebank .dis file')
    parser.add_argument('output_path',
                        help='path to where the converted output should go')
    args = parser.parse_args()

    with open(args.input_path) as input_file:
        with open(args.output_path, 'w') as output_file:
            tree = ParentedTree.fromstring(input_file.read().strip())
            reformat_rst_tree(tree)
            collapse_rst_labels(tree)
            print(tree.pprint(TREE_PRINT_MARGIN), file=output_file)


if __name__ == '__main__':
    main()
