#!/usr/bin/env python

"""
Convert RST discourse tree bank into gold standard sequence of actions.

This script converts the RST discourse treebank into a gold standard
sequence of shift reduce parsing actions.

This is based on Perl code (``trees2actionseq.pl``) from Kenji Sagae.

For each tree in the input, this will output a line representing the
shift-reduce action sequence for that tree.  The actions in a sequence
will be space-separated.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse

from nltk.tree import ParentedTree

from .discourse_parsing import ShiftReduceAction


def extract_parse_actions(tree):
    """
    Extract a list of ``ShiftReduceAction`` objects for the given tree.

    Parameters
    ----------
    tree : nltk.tree.ParentedTree
        The RST tree from which to extract the actions.

    Returns
    -------
    actseq : list
        List of ``ShiftReduceAction`` objects extracted from the tree.
    """
    if tree.label() == '':
        tree.set_label("ROOT")
    assert tree.label() == "ROOT"

    stack = []
    cstack = [ParentedTree.fromstring("(DUMMY0 (DUMMY1 DUMMY3))")]
    actseq = []
    _extract_parse_actions_helper(tree, stack, cstack, actseq)
    actseq = _merge_constituent_end_shifts(actseq)

    return actseq


def _merge_constituent_end_shifts(actseq):
    """
    Remove unnecessary unary reduce action.

    The ``_extract_parse_actions_helper()`` function below always puts a '*'
    on binary reduce actions, and then puts a unary reduce after a sequence of
    binary reduce actions for the same constituent.  This method will remove
    the unary reduce and make the last binary reduce not have a '*', indicating
    that the constituent is complete.

    Parameters
    ----------
    actseq : list
        List of ``ShiftReduceAction`` objects.

    Returns
    -------
    res : list
        Updated list of ``ShiftReduceAction`` objects.
    """
    res = []
    for act in actseq:
        if act.type == 'U' and res and res[-1].type == 'B':
            assert f"{act.label}*" == res[-1].label
            tmp_act = res.pop()
            res.append(ShiftReduceAction(type=tmp_act.type, label=act.label))
        else:
            res.append(act)
    return res


def _is_head_of(node1, node2):
    """
    Check if ``node1`` is the head of ``node2``.

    Parameters
    ----------
    node1 : nltk.tree.ParentedTree
        The first node.
    node2 : nltk.tree.ParentedTree
        The second node.

    Returns
    -------
    is_head : bool
        ``True`` if ``node1`` is the head of ``node2``, ``False`` otherwise.
    """
    node1parent = node1.parent()
    if node2.parent() != node1parent:
        return False

    if node1.label().startswith("nucleus:"):
        # TODO: simplify using or
        if node2.label().startswith("satellite:"):
            return True
        elif node1parent.index(node1) < node1parent.index(node2):
            return True

    return False


def _extract_parse_actions_helper(node, stack, cstack, actseq):
    """
    Helper function for ``extract_parse_actions()``.

    Parameters
    ----------
    node : nltk.tree.ParentedTree
        The input node.
    stack : list
        The complete stack.
    cstack : list
        The current stack pointer.
    actseq : list
        List of ``ShiftReduceAction`` objects where the extracted actions
        for this node will be stored.
    """
    stack.append(node)

    for child in node:
        if isinstance(child, str):
            continue
        _extract_parse_actions_helper(child, stack, cstack, actseq)

    nt = stack.pop()

    # If the current node is a preterminal, add a shift action.
    tmp_parent = cstack[-1].parent()
    if isinstance(nt[0], str):
        actseq.append(ShiftReduceAction(type='S', label="text"))
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
        # then add a ``binary_reduce`` action.

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
                                            label=f"{new_label}*"))
            chflg = True


def main():  # noqa: D103
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mrg_path",
                        help="A file with constituent trees in ``mrg`` format.")
    args = parser.parse_args()

    with open(args.mrg_path) as constituent_file:
        for line in constituent_file:
            tree = ParentedTree.fromstring(line.strip())
            actseq = extract_parse_actions(tree)
            print(" ".join([f"{act.type}:{act.label}" for act in actseq]))


if __name__ == "__main__":
    main()
