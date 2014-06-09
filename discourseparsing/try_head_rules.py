#!/usr/bin/env python

'''
A simple script for testing out head rules on a PTB file.
'''

import argparse
import re

from discourseparsing.tree_util import HeadedParentedTree
from discourseparsing.convert_rst_discourse_tb import convert_ptb_tree

def depth(t):
    res = 0
    parent = t
    while parent.parent() is not None:
        parent = parent.parent()
        res += 1
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ptb_file', help='PTB MRG file')
    args = parser.parse_args()

    with open(args.ptb_file) as f:
        doc = re.sub(r'\s+', ' ', f.read()).strip()
        trees = [HeadedParentedTree('( ({}'.format(x)) for x
                 in re.split(r'\(\s*\(', doc) if x]

        for t in trees:
            convert_ptb_tree(t)
            print("\n\n{}".format(t.pprint()))
            for subtree in t.subtrees():
                print("{}{}\t{}".format(' '.join(['' for x in range(depth(subtree))]), subtree.label(), subtree.head_word()))

if __name__ == '__main__':
    main()
