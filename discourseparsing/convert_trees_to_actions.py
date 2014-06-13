#!/usr/bin/env python

'''
A script for converting the RST discourse treebank into a gold standard
sequence of shift reduce parsing actions.

This is based on perl code (trees2actionseq.pl) from Kenji Sagae.

For each tree in the input, this will output a line representing the
shift-reduce action sequence for that tree.  The actions in a sequence will
be space-separated.
'''

import argparse
from nltk.tree import ParentedTree
from collections import namedtuple
import re


ShiftReduceAction = namedtuple('ShiftReduceAction', ['type', 'label'])


# def _find_rst_heads(tree, index_list, heads):


#     # preterminal (stopping condition)
#     if isinstance(tree[0], str):
#         heads[index] = index
#         return

#     for i, child in enumerate(tree):
#         _find_rst_heads(tree, index, heads)



# def compute_rst_dependencies(tree):
#     dtree = []
#     dtree.append({'idx': 0, 'word': "leftwall", 'pos': "LW", 'link': -1})

#     # Do a post-order traversal of the tree and
#     # set the head of each node to be the head of the leftmost of its children
#     # (or itself if it is a preterminal).
#     _find_rst_heads(tree, [0], {})


#     # For each preterminal, walk up the tree until the current node's head is 
#     # not the preterminal.  Then, set the preterminal's head to be that node's
#     # head.  Add the resulting dependency.

#     for part in parts:
#         a = re.split(r'[ \t]', part)
#         dtree.append({'idx': a[0],
#                       'word': a[1],
#                       'lemma': a[2],
#                       'cpos': a[3],
#                       'pos': a[4],
#                       'morph': a[5],
#                       'link': a[6]})
#     return dtree


def convert_trees_to_actions(constituent_file):
    tree = ""

    for line in constituent_file:
        tree = ParentedTree(line.strip())
        tree.set_label('ROOT')
        #dtree = compute_rst_dependencies(tree)

        stack = []
        cstack = [ParentedTree('(DUMMY0 (DUMMY1 DUMMY3))')]
        #dstack = [0]
        #dptr = [0]  # this is a list so it's mutable and passed by reference

        actseq = []

        _convert_tree_to_actions_helper(tree, stack, cstack, actseq)

        actseq = merge_constituent_end_shifts(actseq)

        yield actseq


def merge_constituent_end_shifts(actseq):
    '''
    convert_tree_to_actions_helper always puts * on binary reduce actions,
    and then puts a unary reduce after a sequence of binary reduces for the
    same constituent.  This method will remove the unary reduce and make the
    last binary reduce not have a *, indicating that the constituent is
    complete.
    '''

    res = []
    for act in actseq:
        if act.type == 'U' and res and (res[-1].type == 'L'
                                        or res[-1].type == 'R'):
            assert '{}*'.format(act.label) == res[-1].label
            tmp_act = res.pop()
            res.append(ShiftReduceAction(type=tmp_act.type, label=act.label))
        else:
            res.append(act)
    return res

def _is_head_of(n1, n2):
    n1parent = n1.parent()
    if n2.parent() != n1parent:
        return False

    if n1.label().startswith('nucleus:'):
        if n2.label().startswith('satellite:'):
            return True
        elif n1parent.index(n1) < n1parent.index(n2):
            return True
    return False


def _convert_tree_to_actions_helper(node, stack, cstack, actseq):
    stack.append(node)

    for child in node:
        if isinstance(child, str):
            continue
        _convert_tree_to_actions_helper(child, stack, cstack, actseq)

    nt = stack.pop()

    # If the current node is a preterminal, add a shift action.
    tmp_parent = cstack[-1].parent()
    if isinstance(nt[0], str):
        actseq.append(ShiftReduceAction(type='S', label='POS'))
        cstack.append(nt)
    # Or if we are at the root of the tree, then add reduce_right:ROOT.
    elif tmp_parent.label() == "ROOT":
        actseq.append(ShiftReduceAction(type='R', label='ROOT'))
        chflg = False
    # Otherwise, we have visited all the children of a nonterminal node,
    # and we should add a unary reduce
    else:
        actseq.append(ShiftReduceAction(type='U', label=tmp_parent.label()))
        cstack.pop()
        cstack.append(nt)

    # Check to see if there should be any reduce right or reduce left actions.
    chflg = True
    while chflg and stack:
        chflg = False
        # If the head of most recently visited node
        # is the 2nd most recently visited node,
        # and they have the same parent,
        # then add a reduce_right.
        if _is_head_of(cstack[-2], cstack[-1]):

            tmpRc = cstack.pop()
            tmpLc = cstack.pop()
            cstack.append(tmpLc)

            actseq.append(ShiftReduceAction(type='R',
                                            label='{}*'.format(tmpLc.parent().label())))
            chflg = True

        # If the most recently visited node
        # is the head of the 2nd most recently visited node
        # and they both have the same parent,
        # then add a reduce_left.
        if _is_head_of(cstack[-1], cstack[-2]):

            tmpRc = cstack.pop()
            tmpLc = cstack.pop()
            cstack.append(tmpRc)

            actseq.append(ShiftReduceAction(type='L',
                                            label='{}*'.format(tmpRc.parent().label())))
            chflg = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mrg_path', help='a file with constituent trees in mrg format.')
    args = parser.parse_args()

    with open(args.mrg_path) as constituent_file:
        for actseq in convert_trees_to_actions(constituent_file):
            print(' '.join(['{}:{}'.format(x.type, x.label) for x in actseq]))


if __name__ == '__main__':
    main()
