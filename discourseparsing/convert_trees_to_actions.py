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
from nltk.tree import Tree
from collections import namedtuple
import re


ShiftReduceAction = namedtuple('ShiftReduceAction', ['type', 'label'])


def get_dependencies(dep_file):
    dtree = []
    dtree.append({'idx': 0, 'word': "leftwall", 'pos': "LW", 'link': -1})

    tmp_str = dep_file.readline().strip()
    while re.search(r'^[ \t\n\r]*$', tmp_str):
        tmp_str = dep_file.readline().strip()

    parts = tmp_str.split('\n')
    for part in parts:
        a = re.split(r'[ \t]', part)
        dtree.append({'idx': a[0],
                      'word': a[1],
                      'lemma': a[2],
                      'cpos': a[3],
                      'pos': a[4],
                      'morph': a[5],
                      'link': a[6]})
    return dtree


def convert_trees_to_actions(constituent_file, dep_file):
    tree = ""

    for line in constituent_file:
        tree = Tree(line.strip())

        dtree = get_dependencies(dep_file)

        stack = []
        cstack = [{'nt': 'LEFTWALL', 'parent': 'LEFTWALL', 'tree': ''}]
        dstack = [0]
        dptr = 0

        actseq = []

        convert_tree_to_actions_helper(tree, stack, cstack, dstack, dptr, dtree, actseq)

        merge_constituent_end_shifts(actseq)

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
        if act.type == 'U' and res and (res[-1] == 'L' or res[-1] == 'R'):
            assert '{}*'.format(act.label) == res[-1].label
            tmp_act = res.pop()
            res.append(ShiftReduceAction(type=tmp_act.type, label=act.label))
        else:
            res.append(act)
    return res


def convert_tree_to_actions_helper(node, stack, cstack, dstack, dptr, dtree, actseq):
    stack.append(node.label())

    for child in node:
        convert_tree_to_actions_helper(child, stack, cstack, dstack, dptr, dtree, actseq)

    nt = stack.pop()

    # if the current node is a preterminal, add a shift
    if isinstance(nt[0], str):
        actseq.append(ShiftReduceAction(type='S', label='POS'))
        cstack.append({'nt': '{}(from shift)'.format(nt),
                       'parent': stack[-1],
                       'tree': '({})'.format(nt)})
        dptr += 1  # TODO ok if this is modified only locally?
        dstack.append(dptr)
    # otherwise (for nonterminals), add a unary reduce or reduce_right:root
    else:
        tmpstr = cstack[-1]['parent']
        if tmpstr == "TOP":
            actseq.append(ShiftReduceAction(type='R', label='ROOT'))
        else:
            actseq.append(ShiftReduceAction(type='U', label=tmpstr))


        tmpc = cstack.pop()
        cstack.append({'nt': '{}(from reduce = {})'.format(tmpc['nt'], tmpc['parent']),
                       'parent': stack[-1],
                       'tree': '({} {})'.format(tmpc['parent'], tmpc['tree'])})

    # check to see if there should be any reduce right or reduce left actions
    chflg = True
    while chflg:
        chflg = False
        # if the head of most recently visited node is the 2nd most recently visited node,
        # and they have the same parent,
        # then add a reduce_right
        if dtree[dstack[-1]]['link'] == dtree[dstack[-2]]['idx'] \
                and cstack[-1]['parent'] == cstack[-2]['parent']:
            tmpR = dstack.pop()
            tmpL = dstack.pop()
            dstack.append(tmpL)

            tmpRc = cstack.pop()
            tmpLc = cstack.pop()
            cstack.append({'parent': stack[-1],
                           'nt': tmpLc['nt'],
                           'tree': '{} {}'.format(tmpLc['tree'], tmpRc['tree'])})
            tmpstr = cstack[-1]['parent']

            actseq.append(ShiftReduceAction(type='R', label='{}*'.format(tmpstr)))

            chflg = True

        # if the most recently visited node is the head of the 2nd most recently visited node
        # and they both have the same parent,
        # then add a reduce_left
        if dtree[dstack[-2]]['link'] == dtree[dstack[-1]]['idx'] \
                and cstack[-1]['parent'] == cstack[-2]['parent']:
            tmpR = dstack.pop()
            tmpL = dstack.pop()
            dstack.append(tmpR)

            tmpRc = cstack.pop()
            tmpLc = cstack.pop()
            cstack.append({'parent': stack[-1],
                           'nt': tmpRc['nt'],
                           'tree': '{} {}'.format(tmpLc['tree'], tmpRc['tree'])})
            tmpstr = cstack[-1]['parent']

            actseq.append(ShiftReduceAction(type='L', label='{}*'.format(tmpstr)))

            chflg = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mrg_path', help='a file with constituent trees in mrg format.')
    parser.add_argument('dep_path', help='the corresponding dependency file in CoNLL-X format.')
    args = parser.parse_args()

    with open(args.mrg_path) as constituent_file, open(args.dep_path) as dep_file:
        for actseq in convert_trees_to_actions(constituent_file, dep_file):
            print(' '.join(actseq))


if __name__ == '__main__':
    main()
