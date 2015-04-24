#!/usr/bin/env python
# License: MIT

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

from discourseparsing.discourse_parsing import ShiftReduceAction


def extract_parse_actions(tree):
    '''
    Extracts a list of ShiftReduceAction objects for the given tree.
    '''
    if tree.label() == '':
        tree.set_label('ROOT')
    assert tree.label() == 'ROOT'

    stack = []
    cstack = [ParentedTree.fromstring('(DUMMY0 (DUMMY1 DUMMY3))')]
    actseq = []
    _extract_parse_actions_helper(tree, stack, cstack, actseq)
    actseq = _merge_constituent_end_shifts(actseq)

    return actseq


def _merge_constituent_end_shifts(actseq):
    '''
    convert_tree_to_actions_helper always puts * on binary reduce actions,
    and then puts a unary reduce after a sequence of binary reduces for the
    same constituent.  This method will remove the unary reduce and make the
    last binary reduce not have a *, indicating that the constituent is
    complete.
    '''

    res = []
    for act in actseq:
        if act.type == 'U' and res and res[-1].type == 'B':
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


def _extract_parse_actions_helper(node, stack, cstack, actseq):
    stack.append(node)

    for child in node:
        if isinstance(child, str):
            continue
        _extract_parse_actions_helper(child, stack, cstack, actseq)

    nt = stack.pop()

    # If the current node is a preterminal, add a shift action.
    tmp_parent = cstack[-1].parent()
    if isinstance(nt[0], str):
        actseq.append(ShiftReduceAction(type='S', label='text'))
        cstack.append(nt)
    # Otherwise, we have visited all the children of a nonterminal node,
    # and we should add a unary reduce
    else:
        actseq.append(ShiftReduceAction(type='U', label=tmp_parent.label()))
        cstack.pop()
        cstack.append(nt)

    # Check to see if there should be any binary reduce actions.
    chflg = True
    while chflg and stack:
        chflg = False
        # If the two most recently visited nodes have the same parent,
        # then add a binary_reduce action.
        # Note that this approach will still work if there are multiple
        # satellite children because the ones nearest to the nucleus will be
        # reduced first, and eventually all the satellites will be binary
        # reduced with the nucleus.
        headR = _is_head_of(cstack[-1], cstack[-2])
        headL = _is_head_of(cstack[-2], cstack[-1])
        if headL or headR:
            tmpRc = cstack.pop()
            tmpLc = cstack.pop()
            if headR:
                # reduce left (right node becomes head)
                cstack.append(tmpRc)
                new_label = tmpRc.parent().label()
            else:
                # reduce right (left node becomes head)
                cstack.append(tmpLc)
                new_label = tmpLc.parent().label()

            actseq.append(ShiftReduceAction(type='B',
                                            label='{}*'.format(new_label)))
            chflg = True


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'mrg_path', help='a file with constituent trees in mrg format.')
    args = parser.parse_args()

    with open(args.mrg_path) as constituent_file:
        for line in constituent_file:
            tree = ParentedTree.fromstring(line.strip())
            actseq = extract_parse_actions(tree)
            print(' '.join(['{}:{}'.format(x.type, x.label) for x in actseq]))


if __name__ == '__main__':
    main()
