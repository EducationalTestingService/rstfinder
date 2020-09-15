"""
Classes and functions pertaining to syntactic (constituency) trees.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""
import re

from nltk.tree import ParentedTree

TREE_PRINT_MARGIN = 1000000000

_ptb_paren_mapping = {'(': r"-LRB-",
                      ')': r"-RRB-",
                      '[': r"-LSB-",
                      ']': r"-RSB-",
                      '{': r"-LCB-",
                      '}': r"-RCB-"}
_reverse_ptb_paren_mapping = {bracket_replacement: bracket_type
                              for bracket_type, bracket_replacement
                              in _ptb_paren_mapping.items()}


class HeadedParentedTree(ParentedTree):
    """
    Modify ``nltk.tree.ParentedTree`` to also return heads.

    This subclass of ``nltk.tree.ParentedTree`` also returns heads
    using head rules from Michael Collins's 1999 thesis, Appendix A. 
    See the ``head()`` method below.
    """

    start_points = {"ADJP": "L",
                    "ADVP": "R",
                    "CONJP": "R",
                    "FRAG": "R",
                    "INTJ": "L",
                    "LST": "R",
                    "NAC": "L",
                    "PP": "R",
                    "PRN": "L",
                    "PRT": "R",
                    "QP": "L",
                    "RRC": "R",
                    "S": "L",
                    "SBAR": "L",
                    "SBARQ": "L",
                    "SINV": "L",
                    "SQ": "L",
                    "UCP": "R",
                    "VP": "L",
                    "WHADJP": "L",
                    "WHADVP": "R",
                    "WHNP": "L",
                    "WHPP": "R",
                    "NX": "L",
                    "X": "L"}

    priority_list = {"ADJP": ["NNS", "QP", "NN", "$", "ADVP", "JJ", "VBN",
                              "VBG", "ADJP", "JJR", "NP", "JJS", "DT", "FW",
                              "RBR", "RBS", "SBAR", "RB"],
                     "ADVP": ["RB", "RBR", "RBS", "FW", "ADVP", "TO", "CD",
                              "JJR", "JJ", "IN", "NP", "JJS", "NN"],
                     "CONJP": ["CC", "RB", "IN"],
                     "FRAG": [],
                     "INTJ": [],
                     "LST": ["LS", ":"],
                     "NAC": ["NN", "NNS", "NNP", "NNPS", "NP", "NAC", "EX",
                             "$", "CD", "QP", "PRP", "VBG", "JJ", "JJS",
                             "JJR", "ADJP", "FW"],
                     "PP": ["IN", "TO", "VBG", "VBN", "RP", "FW"],
                     "PRN": [],
                     "PRT": ["RP"],
                     "QP": ["$", "IN", "NNS", "NN", "JJ", "RB", "DT", "CD",
                            "NCD", "QP", "JJR", "JJS"],
                     "RRC": ["VP", "NP", "ADVP", "ADJP", "PP"],
                     "S": ["TO", "IN", "VP", "S", "SBAR", "ADJP", "UCP", "NP"],
                     "SBAR": ["WHNP", "WHPP", "WHADVP", "WHADJP", "IN", "DT",
                              "S", "SQ", "SINV", "SBAR", "FRAG"],
                     "SBARQ": ["SQ", "S", "SINV", "SBARQ", "FRAG"],
                     "SINV": ["VBZ", "VBD", "VBP", "VB", "MD", "VP", "S",
                              "SINV", "ADJP", "NP"],
                     "SQ": ["VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ"],
                     "UCP": [],
                     "VP": ["TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG",
                            "VBP", "VP", "ADJP", "NN", "NNS", "NP"],
                     "WHADJP": ["CC", "WRB", "JJ", "ADJP"],
                     "WHADVP": ["CC", "WRB"],
                     "WHNP": ["WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP"],
                     "WHPP": ["IN", "TO", "FW"],
                     "NX": [],
                     "X": []}

    def __init__(self, node_or_str, children=None):
        """Initialize the tree."""
        self._head = None
        super(HeadedParentedTree, self).__init__(node_or_str, children)

    def _search_children(self, search_list, start_point):
        """
        Find heads of noun phrases.

        This is a helper function for finding heads of noun phrases.
        It finds the first node whose label is in search_list, starting
        from start_point, either "L" for left (i.e., 0) or "R" for right.

        Parameters
        ----------
        search_list : list
            List of labels.
        start_point : str
            The starting point for the search.

        Returns
        -------
        head_index : int
            The positional index of the head node.
        """
        assert start_point == "L" or start_point == "R"

        head_index = None
        num_children = len(self)
        children = list(self)

        # reverse the list if we start from the right
        if start_point == "R":
            children.reverse()

        for i, child in enumerate(children):
            if child.label() in search_list:
                head_index = i
                break

        # correct the index if we reversed the list to start from the right
        if start_point == "R" and head_index is not None:
            head_index = num_children - 1 - head_index

        return head_index

    def head(self):
        """
        Find the head of the tree.

        This method uses the head finding rules, following Michael Collins'
        head rules (from his 1999 Ph.D. thesis, Appendix A).

        A default of the leftmost child was added for NX nodes, which aren't
        discussed in Collin's thesis.  This follows the Stanford Parser
        (http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/trees/CollinsHeadFinder.html).
        """
        if self._head is None:
            num_children = len(self)
            head_index = None

            if num_children < 2:
                # shortcut for when there is only one child
                self._head = self[0]
                return self._head

            # special case: NPs
            if self.label() == 'NP':
                # If last node is POS, that's the head
                if self[-1].label() == "POS":
                    head_index = num_children - 1

                # Otherwise, look right to left for NN, NNP, NNPS, NNS, NX,
                # POS, or JJR.
                if head_index is None:
                    head_index = self._search_children(["NN", "NNP", "NNPS",
                                                        "NNS", "NX", "POS",
                                                        "JJR"],
                                                       "R")

                # Otherwise, search left to right for NP.
                if head_index is None:
                    head_index = self._search_children(["NP"],
                                                       "L")

                # Otherwise, search right to left for $, ADJP, PRN.
                if head_index is None:
                    head_index = self._search_children(["$", "ADJP", "PRN"],
                                                       "R")

                # Otherwise, search right to left for CD.
                if head_index is None:
                    head_index = self._search_children(["CD"],
                                                       "R")

                # Otherwise, search right to left for JJ, JJS, RB, or QP.
                if head_index is None:
                    head_index = self._search_children(["JJ", "JJS", "RB",
                                                        "QP"],
                                                       "R")

                # Otherwise, return the last child.
                if head_index is None:
                    head_index = num_children - 1

            else:  # typical cases
                start_point = self.start_points[self.label()]

                # Try looking for each symbol in the priority list.
                # Stop at the first match.
                for symbol in self.priority_list[self.label()]:
                    head_index = self._search_children([symbol], start_point)
                    if head_index is not None:
                        break

                if head_index is None:
                    # If none of the symbols given in the priority list
                    # for this label was found, then default to the first
                    # child from the left or right, as specified by the
                    # starting points table.
                    head_index = 0 if start_point == 'L' else num_children - 1

            # special case: coordination.
            # After finding the head, check to see if its left sibling is a
            # conjunction.  If so, move the head index left 2.
            if 'CC' in {x.label() for x in self}:
                if head_index > 2 and self[head_index - 1].label() == 'CC':
                    head_index -= 2

            # cache the result
            self._head = self[head_index]

        return self._head

    def find_maximal_head_node(self):
        """
        Find the topmost node that has this node as its head.

        Returns itself if the parent has a different head.
        """
        res = self
        parent = res.parent()
        while parent is not None and parent.head() == res:
            res = parent
            parent = res.parent()

        return res

    def head_preterminal(self):
        """Return the head preterminal."""
        res = self
        while not isinstance(res[0], str):
            res = res.head()
        return res

    def head_word(self):
        """Return the head word."""
        return self.head_preterminal()[0]

    def head_pos(self):
        """Return the part-of-speech for the head."""
        return self.head_preterminal().label()


def extract_preterminals(tree):
    """Extract the preterminals from the given tree."""
    return [node for node in tree.subtrees() if node.height() == 2]


def convert_paren_tokens_to_ptb_format(toks):
    """Convert parentheses tokens in given list to PTB format."""
    return [_ptb_paren_mapping.get(tok, tok) for tok in toks]


def convert_parens_to_ptb_format(sent):
    """Convert parentheses in given string to PTB format."""
    for key, val in _ptb_paren_mapping.items():
        sent = sent.replace(key, f" {val} ")

    # remove extra spaces added by normalizing brackets
    sent = re.sub(r'\s+', r' ', sent).strip()
    return sent


def extract_converted_terminals(tree):
    """Extract all converted terminals in tree, e.g., parentheses & quotes."""
    res = []
    prev_w = ""
    for w in tree.leaves():
        if prev_w and prev_w == "U.S." and w == '.':
            continue
        if w in _reverse_ptb_paren_mapping:
            w = _reverse_ptb_paren_mapping[w]
        elif w == '``' or w == "''":
            w = '"'

        prev_w = w
        res.append(w)
    return res


def convert_ptb_tree(tree):
    """Convert PTB tree to remove traces etc."""
    for subtree in [st for st in
                    tree.subtrees(filter=lambda x: x.label() == "-NONE-")]:
        curtree = subtree
        while curtree.label() == "-NONE-" or len(curtree) == 0:
            parent = curtree.parent()
            parent.remove(curtree)
            curtree = parent

    # remove suffixes that don't appear in typical parser output
    # (e.g., "-SBJ-1" in "NP-SBJ-1"); leave labels starting with
    # "-" as is (e.g., "-LRB-").
    for subtree in tree.subtrees():
        label = subtree.label()
        if '-' in label and label[0] != '-':
            subtree.set_label(label[:label.index('-')])
        label = subtree.label()
        if '=' in label and label[0] != '=':
            subtree.set_label(label[:label.index('=')])

    # remove escape sequences from words (e.g., "3\\/4")
    for subtree in tree.subtrees():
        if isinstance(subtree[0], str):
            for i in range(len(subtree)):
                subtree[i] = re.sub(r'\\', r'', subtree[i])


def find_first_common_ancestor(node1, node2):
    """
    Find the first common ancestor for two nodes in the same tree.

    Parameters
    ----------
    node1 : nltk.tree.ParentedTree
        The first node.
    node2 : nltk.tree.ParentedTree
        The second node.

    Returns
    -------
    ancestor_node : nltk.tree.ParentedTree
        The first common ancestor for two nodes in the same tree.
    """
    # TODO: write a unit test for this

    # make sure we are in the same tree
    assert node1.root() == node2.root()

    # make a set of all ancestors of node1
    node1_ancestor_treepositions = set()
    node1_parent = node1.parent()
    while node1_parent is not None:
        # note: storing treepositions isn't particularly efficient since
        # treeposition() walks up the tree; using memory addresses like
        # id(node1_parent) would be faster, but seems potentially
        # hazardous/confusing
        node1_ancestor_treepositions.add(node1_parent.treeposition())
        node1_parent = node1_parent.parent()

    # find the first ancestor of node2 that is also an ancestor of node1
    node2_parent = node2.parent()
    res = None
    while node2_parent is not None:
        if node2_parent.treeposition() in node1_ancestor_treepositions:
            res = node2_parent
            break
        node2_parent = node2_parent.parent()

    assert res is not None
    return res


def collapse_binarized_nodes(tree):
    """
    Remove temporary, binarized nodes from the given tree.

    For each node that is marked as a temporary, binarized node (with a *),
    remove it from its parent and add its children in its place.

    Note that this modifies the tree in place.
    """
    assert isinstance(tree, ParentedTree)

    # TODO: write a unit test for this method
    to_process = []
    for subtree in tree.subtrees():
        to_process.append(subtree)

    # do a reverse of the pre-order traversal implicit
    # in the subtrees methods, so the leaves are visited first.
    for subtree in reversed(to_process):
        if subtree.label().endswith('*'):
            parent = subtree.parent()
            assert (subtree.label() == parent.label() or
                    subtree.label()[:-1] == parent.label())
            tmp_index = parent.index(subtree)
            del parent[tmp_index]
            while subtree:
                child = subtree.pop()
                parent.insert(tmp_index, child)

    # make sure the output is correct
    for subtree in tree.subtrees():
        assert not subtree.label().endswith('*')
