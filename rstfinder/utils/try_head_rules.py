#!/usr/bin/env python
"""
Test out head rules on a Penn Treebank file.

A simple script for testing out head rules on a PTB file.
"""

import argparse
import re

from rstfinder.convert_rst_discourse_tb import convert_ptb_tree
from rstfinder.tree_util import HeadedParentedTree


def depth(tree):
    """Compute the depth of the given tree."""
    res = 0
    parent = tree
    while parent.parent() is not None:
        parent = parent.parent()
        res += 1
    return res


def main():  # noqa: D103
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ptb_file", help="PTB MRG file")
    args = parser.parse_args()

    with open(args.ptb_file) as ptbfh:
        doc = re.sub(r'\s+', ' ', ptbfh.read()).strip()
        trees = [HeadedParentedTree.fromstring(f"( ({x}") for x
                 in re.split(r'\(\s*\(', doc) if x]

        for tree in trees:
            convert_ptb_tree(tree)
            print("\n\n{}".format(tree.pformat()))
            for subtree in tree.subtrees():
                subtree_label = subtree.label()
                subtree_head_word = subtree.head_word()
                indentation = ' '.join(['' for x in range(depth(subtree))])
                print(f"{indentation}{subtree_label}\t{subtree_head_word}")


if __name__ == '__main__':
    main()
