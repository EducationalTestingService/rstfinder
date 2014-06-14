#!/usr/bin/env python3

'''
Script to collapse RST discourse treebank relation types,
based on a perl script by Kenji Sagae.
'''

import argparse
import re

from nltk.tree import Tree


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
    res = label
    label_lc = label.lower()
    if re.search(r'^attribution', label_lc):
        res = "ATTRIBUTION"
    elif re.search(r'^(background|circumstance)', label_lc):
        res = "BACKGROUND"
    elif re.search(r'^(cause|result|consequence)', label_lc):
        res = "CAUSE"
    elif re.search(r'^(comparison|preference|analogy|proportion)', label_lc):
        res = "COMPARISON"
    elif re.search(r'^(condition|hypothetical|contingency|otherwise)', label_lc):
        res = "CONDITION"
    elif re.search(r'^(contrast|concession|antithesis)', label_lc):
        res = "CONTRAST"
    elif re.search(r'^(elaboration.*|example|definition)', label_lc):
        res = "ELABORATION"
    elif re.search(r'^(purpose|enablement)', label_lc):
        res = "ENABLEMENT"
    elif re.search(r'^(evaluation|interpretation|conclusion|comment)', label_lc):
        res = "EVALUATION"
    elif re.search(r'^(evidence|explanation.*|reason)', label_lc):
        res = "EXPLANATION"
    elif re.search(r'^(list|disjunction)', label_lc):
        res = "JOINT"
    elif re.search(r'^(manner|means)', label_lc):
        res = "MANNERMEANS"
    elif re.search(r'^(problem\-solution|question\-answer|statement\-response|topic\-comment|comment\-topic|rhetorical\-question)', label_lc):
        res = "TOPICCOMMENT"
    elif re.search(r'^(summary|restatement)', label_lc):
        res = "SUMMARY"
    elif re.search(r'^(temporal\-.*|sequence|inverted\-sequence)', label_lc):
        res = "TEMPORAL"
    elif re.search(r'^(topic-.*)', label_lc):
        res = "TOPICCHANGE"
    res = res.lower()
    # elif re.search(r'^(span|same\-unit|textualorganization)', label_lc):
    #    res = label

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_path', help='path to an original RST discourse treebank .dis file')
    parser.add_argument('output_path',
                        help='path to where the converted output should go')
    args = parser.parse_args()

    with open(args.input_path) as input_file:
        with open(args.output_path, 'w') as output_file:
            for line in input_file:
                tree = Tree(line.strip())
                collapse_rst_labels(tree)
                print(re.sub(r'\s+', r' ', str(tree)), file=output_file)


if __name__ == '__main__':
    main()
