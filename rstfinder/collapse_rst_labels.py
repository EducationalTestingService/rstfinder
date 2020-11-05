#!/usr/bin/env python

"""
Collapse RST discourse treebank relation types.

This script is based on a Perl script by Kenji Sagae.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import re

from nltk.tree import ParentedTree

from .reformat_rst_trees import reformat_rst_tree
from .tree_util import TREE_PRINT_MARGIN


def collapse_rst_labels(tree):
    """
    Collapse the RST labels to a smaller set.

    This function collapses the RST labels to the set of 18 described
    by the Carlson et al. paper that comes with the RST discourse treebank.

    **IMPORTANT**: The input tree is modified tree in place.

    Parameters
    ----------
    tree : nltk.tree.ParentedTree
        The input tree for which to collapse the labels.
    """
    # walk the tree, and collapse the labels for each subtree
    for subtree in tree.subtrees():
        subtree.set_label(_collapse_rst_label(subtree.label()))


def _collapse_rst_label(label):
    """
    Collapse the given label to a smaller set.

    Parameters
    ----------
    label : str
        The label to be collapsed.

    Returns
    -------
    collapsed_label : str
        The collapsed label.

    Raises
    ------
    ValueError
        If the relation type in the input label is unknown.
    """
    if not re.search(':', label):
        return label

    # split the input label into direction and relation
    direction, relation = label.split(':')

    # lowercase the relation before collapsing
    relation_uncased = relation.lower()

    # go through the various relation types and collapse them as needed
    if re.search(r'^attribution', relation_uncased):
        relation = "ATTRIBUTION"

    elif re.search(r'^(background|circumstance)', relation_uncased):
        relation = "BACKGROUND"

    elif re.search(r'^(cause|result|consequence)', relation_uncased):
        relation = "CAUSE"

    elif re.search(r'^(comparison|preference|analogy|proportion)',
                   relation_uncased):
        relation = "COMPARISON"

    elif re.search(r'^(condition|hypothetical|contingency|otherwise)',
                   relation_uncased):
        relation = "CONDITION"

    elif re.search(r'^(contrast|concession|antithesis)', relation_uncased):
        relation = "CONTRAST"

    elif re.search(r'^(elaboration.*|example|definition)', relation_uncased):
        relation = "ELABORATION"

    elif re.search(r'^(purpose|enablement)', relation_uncased):
        relation = "ENABLEMENT"

    elif re.search(r'^(problem\-solution|question\-answer|statement\-response|topic\-comment|comment\-topic|rhetorical\-question)', relation_uncased):
        relation = "TOPICCOMMENT"

    elif re.search(r'^(evaluation|interpretation|conclusion|comment)',
                   relation_uncased):
        # note that this check for "comment" needs to come after the one
        # above that looks for "comment-topic"
        relation = "EVALUATION"

    elif re.search(r'^(evidence|explanation.*|reason)', relation_uncased):
        relation = "EXPLANATION"

    elif re.search(r'^(list|disjunction)', relation_uncased):
        relation = "JOINT"

    elif re.search(r'^(manner|means)', relation_uncased):
        relation = "MANNERMEANS"

    elif re.search(r'^(summary|restatement)', relation_uncased):
        relation = "SUMMARY"

    elif re.search(r'^(temporal\-.*|sequence|inverted\-sequence)',
                   relation_uncased):
        relation = "TEMPORAL"

    elif re.search(r'^(topic-.*)', relation_uncased):
        relation = "TOPICCHANGE"

    elif re.search(r'^(span|same\-unit|textualorganization)$', relation_uncased):
        pass

    else:
        raise ValueError(f"unknown relation type in label: {label}")

    # TODO: make this all upper case (to resemble PTB nonterminals)?
    collapsed_label = f"{direction}:{relation}".lower()
    return collapsed_label


def main():  # noqa: D103
    """
    Main function.

    Args:
    """
    parser = argparse.ArgumentParser(description="Note that this main "
                                                 "method is just for testing.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_path",
                        help="Path to an RST discourse treebank .dis file.")
    parser.add_argument("output_path",
                        help="Path to file containing the collapsed output.")
    args = parser.parse_args()

    # open the input file containing each input tree on a single line,
    # collapse it, and then print it out to the given output file
    with open(args.input_path, 'r') as input_file,\
            open(args.output_path, 'w') as output_file:
        tree = ParentedTree.fromstring(input_file.read().strip())
        reformat_rst_tree(tree)
        collapse_rst_labels(tree)
        tree.pprint(margin=TREE_PRINT_MARGIN, file=output_file)


if __name__ == "__main__":
    main()
